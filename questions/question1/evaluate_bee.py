# 解析参数
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from questions.question1.bee_tool import encode, writeVectors, predict, removeFiles, readWords, start_nlp


def load_dataset():

    dataset = pd.read_pickle('./all_statement_aug.pkl')
    X_train, X_eval, y_train, y_eval = train_test_split(dataset['text'], dataset['labels'].apply(lambda x: eval(x)),
                                                            test_size=0.3, random_state=42)
    X_test, X_eval, y_test, y_eval = train_test_split(X_eval, y_eval, test_size=0.5, random_state=42)

    return X_eval, y_eval


def predict_sent(sentences, stanford_nlp):
    requestCounter = "test_data"
    try:
        # ecode the sentences
        encodeResult = encode(sentences, sentences, stanford_nlp)
        sentVectors = encodeResult["sentVectors"]

        # write the sentences to a file
        inputFileName = writeVectors(sentVectors, requestCounter)

        # predict the output
        obPrediction = predict("OB", inputFileName, requestCounter)
        ebPrediction = predict("EB", inputFileName, requestCounter)
        srPrediction = predict("SR", inputFileName, requestCounter)

        result = [obPrediction, ebPrediction, srPrediction]
        rst = []
        for item in result:
            rst.append([1 if float(i) > 0 else 0 for i in item ])
        return rst

    except Exception as err:
        print('There was an error: ' + str(err))
        raise err
    finally:
        removeFiles(requestCounter)


def evaluate_metric(truth_label, predict_label):

    truth_label = [list(i) for i in truth_label]

    truth_label, predict_label = pd.DataFrame(truth_label), pd.DataFrame(predict_label)
    """计算预测结果"""
    num_labels = truth_label.shape[1]  # 获取标签的数量

    precision_scores, recall_scores, accuracy_scores, f1_scores = [], [], [], []

    for label_idx in range(num_labels):
        # 计算每一个标签的评价指标
        label_preds = predict_label.iloc[:, label_idx]
        label_labels = truth_label.iloc[:, label_idx]

        # 计算 TP, FP, FN, TN
        TP = np.sum((label_preds == 1) & (label_labels == 1))
        FP = np.sum((label_preds == 1) & (label_labels == 0))
        FN = np.sum((label_preds == 0) & (label_labels == 1))
        TN = np.sum((label_preds == 0) & (label_labels == 0))

        # 计算 Precision, Recall, Accuracy, F1 Score
        precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        recall = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        f1_score = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

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
            f"====label {idx+1} metrics====\nprecision: {precision_scores[idx]:.4f}, "
            f"recall: {recall_scores[idx]:.4f}, acc: {accuracy_scores[idx]:.4f}, F1: {f1_scores[idx]:.4f}"
        )

    print(
        f"======evaluation average metrics======\n"
        f"precision: {avg_precision:.4f}, average recall: {avg_recall:.4f}, "
        f"average acc: {avg_accuracy:.4f}, average F1: {avg_f1:.4f}"
    )


if __name__ == '__main__':

    test_data, test_label = load_dataset()
    print(len(test_label))
    readWords()
    core_nlp = start_nlp()
    pre_label = predict_sent(test_data.tolist(), core_nlp)
    evaluate_metric(test_label, pre_label)
    core_nlp.close()
