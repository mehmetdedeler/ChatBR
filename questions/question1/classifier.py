# 导入必要的库
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, \
    BertConfig
import torch.nn.functional as F


# 定义数据集类，将文本转化为BERT输入
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts.to_numpy()
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }


class FocalLoss(nn.Module):
    def __init__(self, alpha=torch.tensor([0.3, 0.5, 0.3]), gamma=2, device='cpu'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha.to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 计算二分类交叉熵损失
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # 计算预测概率
        pt = torch.exp(-bce_loss)
        # 计算Focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def parse_arguments():
    """解析参数"""
    parser = argparse.ArgumentParser(description='Eval Arguments.')
    parser.add_argument('--batch_size', type=int, default=128, help='Paradigm to use')
    parser.add_argument('--epoch_num', type=int, default=10, help='Paradigm to use')
    parser.add_argument('--learning_rate', default=1e-5, help='Paradigm to use')
    parser.add_argument('--max_length', default=128)
    parser.add_argument('--num_labels', default=3)
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--hidden_dropout_prob', default=0.3)
    parser.add_argument('--data_path', default='./sent_data.pkl')
    parser.add_argument('--model_path', default='./model/bert-base')
    parser.add_argument('--model_save_path', default="./model/bert-tuning-no-da")
    parser.add_argument('--class_weights', default=None)
    parser.add_argument('--argument', choices=['no', 'smote', 'class_weight'], default='no')

    arguments = parser.parse_args()

    return arguments


def create_model(args):
    # 实例化BERT分词器和模型
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    config = BertConfig.from_pretrained(args.model_path, num_labels=args.num_labels,
                                        hidden_dropout_prob=args.hidden_dropout_prob)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)

    # 将模型移动到合适的设备上（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    args.device = device

    return model, tokenizer


def calculate_metrics(preds, labels, args):
    """计算预测结果"""
    num_labels = labels.shape[1]  # 获取标签的数量

    precision_scores, recall_scores, accuracy_scores, f1_scores = [], [], [], []

    for label_idx in range(num_labels):
        # 计算每一个标签的评价指标
        label_preds = preds[:, label_idx]
        label_labels = labels[:, label_idx]

        # 计算 TP, FP, FN, TN
        TP = np.sum((label_preds == 1) & (label_labels == 1))
        FP = np.sum((label_preds == 1) & (label_labels == 0))
        FN = np.sum((label_preds == 0) & (label_labels == 1))
        TN = np.sum((label_preds == 0) & (label_labels == 0))

        # 计算 Precision, Recall, Accuracy, F1 Score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 保存结果
        precision_scores.append(precision)
        recall_scores.append(recall)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1_score)

    # 计算所有标签的平均指标
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)

    # 打印每个标签的指标和平均指标
    # labels_name = ['OB', 'EB', 'SR']
    for idx in range(len(precision_scores)):
        print(
            f"====label {idx + 1} metrics====\nprecision: {precision_scores[idx]:.4f}, "
            f"recall: {recall_scores[idx]:.4f}, acc: {accuracy_scores[idx]:.4f}, F1: {f1_scores[idx]:.4f}"
        )

    print(
        f"======evaluation average metrics======\n"
        f"precision: {avg_precision:.4f}, average recall: {avg_recall:.4f}, "
        f"average acc: {avg_accuracy:.4f}, average F1: {avg_f1:.4f}"
    )


def eval_model(model, data_loader, loss_fn, device):
    model.eval()

    predict_array, label_array, loss_list = np.empty((0, 3)), np.empty((0, 3)), []

    with torch.no_grad():
        p_bar = tqdm(data_loader, total=len(data_loader))
        for idx, batch in enumerate(p_bar):
            # 处理input_ids, attention_mask label
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # 输入模型
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 计算loss
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss_list.append(loss.item())

            # 保存预测值和真实值
            preds = torch.sigmoid(logits).round().detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            predict_array = np.vstack((predict_array, preds))
            label_array = np.vstack((label_array, labels))

            p_bar.set_description('eval batch:{}, loss:{:.4f}'.format(idx, loss.item()))
    # 计算指标
    calculate_metrics(predict_array, label_array, args)

    return sum(loss_list) / len(loss_list)


def test_model(args, data_loader):
    # 实例化BERT分词器和模型
    tokenizer = BertTokenizer.from_pretrained(args.model_save_path)
    model_config = BertConfig.from_pretrained(args.model_save_path, num_labels=args.num_labels,
                                              hidden_dropout_prob=args.hidden_dropout_prob)
    model = BertForSequenceClassification.from_pretrained(args.model_save_path, config=model_config)

    # 将模型移动到合适的设备上（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    args.device = device

    model.eval()

    predict_array, label_array, loss_list = np.empty((0, 3)), np.empty((0, 3)), []

    loss_fn = nn.BCEWithLogitsLoss(weight=args.class_weights).to(device)

    with torch.no_grad():
        p_bar = tqdm(data_loader, total=len(data_loader))
        for idx, batch in enumerate(p_bar):
            # 处理input_ids, attention_mask label
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # 输入模型
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 计算loss
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss_list.append(loss.item())

            # 保存预测值和真实值
            preds = torch.sigmoid(logits).round().detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            predict_array = np.vstack((predict_array, preds))
            label_array = np.vstack((label_array, labels))

            p_bar.set_description('eval batch:{}, loss:{:.4f}'.format(idx, loss.item()))
    # 计算指标
    calculate_metrics(predict_array, label_array, args)


# 定义一个函数，训练模型，并在每个epoch结束时在验证集上评估模型
def train_model(model, train_data_loader, eval_data_loader, args):
    # 初始化early stopping的相关变量
    best_val_loss, patience, early_stopping_counter = float("inf"), 1, 0

    device = args.device

    # 定义优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_data_loader) * args.epoch_num * args.batch_size
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 定义损失函数，这里使用二元交叉熵损失，因为是多标签分类任务
    loss_fn = nn.BCEWithLogitsLoss(weight=args.class_weights).to(device)
    # loss_fn = focal_loss

    # 训练模型
    print("====start training model====")
    model.train()
    for epoch in range(args.epoch_num):
        print(f'===Epoch {epoch + 1}/{args.epoch_num} training===')
        loss_list = []
        torch.cuda.empty_cache()  # 清空cuda缓存
        p_bar = tqdm(train_data_loader, total=len(train_data_loader))
        for idx, batch in enumerate(p_bar):
            # text = pd.DataFrame(batch['text'])
            # text.to_csv('./text/{}.csv'.format(idx))

            optimizer.zero_grad()

            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # 计算loss
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            # 向后传播
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            # 记录loss
            loss_list.append(loss.item())

            p_bar.set_description('epoch:{}, batch:{}, loss:{:.4f}'.format(epoch, idx, loss.item()))
        print(f'===end epoch {epoch + 1}/{args.epoch_num} training, mean loss: {np.mean(loss_list):.4f}===')
        # eval
        print(f'===Epoch {epoch + 1}/{args.epoch_num} evaluating===')
        avg_val_loss = eval_model(model, eval_data_loader, loss_fn, device)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= patience:  # 满足早停条件
            break
    # save model
    model.save_pretrained(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)


def load_dataset(tokenizer, args):
    X_test, y_test = None, None
    print("====loading data====")

    dataset = pd.read_pickle(args.data_path)

    if args.argument != 'no':
        # 计算每个类别样本数,并计算权重
        num_samples_per_class = dataset['labels'].value_counts()
        # 将label转为onehot编码
        labels = F.one_hot(torch.tensor(dataset['labels'].apply(lambda x: eval(x))))
        # 划分数据集为训练集和测试集
        X_train, X_eval, y_train, y_eval = train_test_split(dataset['text'], labels,
                                                            test_size=0.2, random_state=42)
        if args.argument == 'class_weight':
            class_weights = 1.0 / torch.tensor(num_samples_per_class, dtype=torch.float)
            args.class_weights = class_weights / class_weights.sum() * len(class_weights)
        elif args.argument == 'smote':  # 使用 SMOTE 进行过采样
            # 划分数据集为训练集和测试集
            X_train, y_train = X_train.numpy(), y_train.numpy()
            # smote
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    # 直接切分数据集
    else:
        X_train, X_eval, y_train, y_eval = train_test_split(dataset['text'],
                                                            torch.FloatTensor(
                                                                dataset['labels'].apply(lambda x: eval(x))),
                                                            test_size=0.3, random_state=42)

        X_test, X_eval, y_test, y_eval = train_test_split(X_eval,
                                                          y_eval,
                                                          test_size=0.5, random_state=42)

    # 创建 DataLoader
    train_dataset = TextDataset(X_train, y_train, tokenizer, args.max_length)
    eval_dataset = TextDataset(X_eval, y_eval, tokenizer, args.max_length)
    test_dataset = TextDataset(X_test, y_test, tokenizer, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    return train_loader, eval_loader, test_loader


if __name__ == '__main__':
    # 解析参数
    args = parse_arguments()
    # 加载模型
    model, tokenizer = create_model(args)
    # 加载数据集
    train_data_loader, eval_data_loader, test_data_loader = load_dataset(tokenizer, args)
    # 调用训练函数，开始训练模型
    # focal_loss = FocalLoss()
    train_model(model, train_data_loader, eval_data_loader, args)

    test_model(args, test_data_loader)
