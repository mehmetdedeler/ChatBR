import argparse
import json
import os
import re

import torch
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

# Import BEE-tool components
import sys
sys.path.append('../question1')
from bee_tool import processText, start_nlp, close_nlp, readWords

# Global variable for Stanford NLP
stanford_nlp = None

def bert_parse_arguments():
    """解析参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', default=128)
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--model_path', default='../question1/model/bert-tuning-01')
    parser.add_argument('--num_labels', default=3)
    parser.add_argument('--hidden_dropout_prob', default=0.3)
    parser.add_argument('--json_bug_path', default='./origin_data')
    parser.add_argument('--bert_result_path', default='./predict_data/bert_new/')
    arguments = parser.parse_args()

    return arguments


def create_bert_model(args):
    """Create BEE-tool model instead of BERT model"""
    global stanford_nlp
    
    # Initialize Stanford NLP
    stanford_nlp = start_nlp()
    
    # Read BEE-tool dictionary
    readWords()
    
    # Create dummy model and tokenizer for compatibility
    class DummyModel:
        def __init__(self):
            self.device = torch.device('cpu')
        
        def to(self, device):
            self.device = device
            return self
        
        def eval(self):
            pass
    
    class DummyTokenizer:
        def __init__(self):
            pass
    
    model = DummyModel()
    tokenizer = DummyTokenizer()
    
    return model, tokenizer


def isEmptySentence(text):
    # 判断语句是否为空
    if len(text) <= 2:
        new_sent = re.sub(r'[*#\->_]|\[\s+\]|&nbsp;|\s{2,}', '', text)
        if len(new_sent) <= 2:
            return True
        else:
            return False
    return False


def parseReportSentences(text):
    # 解析和切分bug report语句
    modifiedSentences = []

    if isEmptySentence(text):
        return modifiedSentences

    # 删除反引号包裹的代码块
    cleaned_text = re.sub(r'```(.*?)```|[*#>_-]+|\[\s+\]|&nbsp;|\s{2,}', '', text, flags=re.DOTALL)
    # 首次切分文本为语句
    sentence_list = re.split(r'[\n\?!]', cleaned_text)
    for sentence in sentence_list:
        # 切分文本成语句
        sub_sentences = sent_tokenize(sentence)
        # 对每个句子进行文本清理和分句
        for sub_sentence in sub_sentences:
            if not isEmptySentence(sub_sentence):
                modifiedSentences.append(sub_sentence.strip())

    return modifiedSentences


def parseSentences(text):
    # 检查文本是否为空，如果为空，就返回一个空列表
    if isEmptySentence(text):
        return []
    # 按照```切分字符串，将被```包裹的所有字符串保存到一个列表里，将剩余的字符串保存到另一个列表里
    sentences = text.split('```')
    modifiedSentences, codeSentences, textSentences = [], [], []
    for i, sentence in enumerate(sentences):
        # 遍历textList列表，对于每个字符串，用sent_tokenize函数将它分割成多个句子，
        # 然后用re.sub函数去掉一些无关的符号，将每个句子加入到结果列表modifiedSentences中
        if not isEmptySentence(sentence):
            if i % 2 == 0:  # 如果是偶数索引，说明是非代码段的字符串
                textSentences = re.split(r'[\n\?\!]', sentence)
                for textSentence in textSentences:
                    if not isEmptySentence(textSentence):
                        for sent in sent_tokenize(textSentence):  # 用sent_tokenize函数将字符串分割成多个句子
                            new_sent = re.sub(r'[*#>_-]+|\[\s+\]|&nbsp;|\s{2,}', '', sent, flags=re.DOTALL)
                            if not isEmptySentence(new_sent):
                                modifiedSentences.append(new_sent.strip())
                    else:
                        continue
            else:
                modifiedSentences.append(sentence)
        else:
            continue

    return modifiedSentences


def predict_and_format(report, model, tokenizer, args):
    """使用BEE-tool分类，并对结果进行处理"""
    report_label_set = []
    new_report = {
        "id": report["bug_id"],
        "title": report['title'],
        "description": "",
        "OB": "",
        "EB": "",
        "SR": ""
    }

    # 遍历缺陷报告的每一条语句
    report_sentence_list = parseReportSentences(report['title'] + "\n" + report['description'])
    for sent_idx, sentence in enumerate(report_sentence_list):
        # 调用BEE-tool模型
        sent_labels = predict(sentence, model, tokenizer, args)
        report_label_set.extend(sent_labels)

        # 重构缺陷报告
        if len(sent_labels):
            for label in sent_labels:
                if label == "OB":
                    new_report["OB"] = new_report["OB"] + sentence + " "
                elif label == "EB":
                    new_report["EB"] = new_report["EB"] + sentence + " "
                else:
                    new_report["SR"] = new_report["SR"] + sentence + " "
        else:
            new_report["description"] = new_report["description"] + sentence + " "

    # 保存含有OB, EB, SR的缺陷报告
    if len(set(report_label_set)) == 3:  # OB, EB, SR
        with open(os.path.join(args.bert_project_result_path, "perfect_data", report["bug_id"] + ".json"), 'w') as f:
            json.dump(new_report, f)
        f.close()
        return 1
    # 保存一般的缺陷报告
    else:
        args.bert_project_path = args.bert_project_result_path  # Add missing variable
        with open(os.path.join(args.bert_project_path, report["bug_id"] + ".json"), 'w') as f:
            json.dump(new_report, f)
        f.close()
        return 0


def predict(sentence, model, tokenizer, args):
    """Use BEE-tool for sentence classification"""
    global stanford_nlp
    
    try:
        # Use BEE-tool to classify the sentence
        result = processText("temp_id", sentence, stanford_nlp)
        
        # Extract labels from the result
        labels = []
        for sent_data in result['bug_report'].values():
            labels.extend(sent_data['labels'])
        
        return labels
    except Exception as e:
        print(f"Error in BEE-tool prediction: {e}")
        return []


def is_report_perfect(report, args):
    """判断一个report是否包含OB, EB,SR"""

    sentence_list = parseSentences(report)

    # 遍历缺陷报告中的每一条语句，并调用BEE-tool模型进行分类
    report_label_set = []
    for sentence in sentence_list:
        sent_labels = predict(sentence, args.model, args.tokenizer, args)
        report_label_set.extend(sent_labels)

    # 判断缺陷报告中是否含有OB, EB,SR
    return set(report_label_set)


def classify_report_quality(model, tokenizer, report, args):
    sentence_list = parseSentences(report)
    # 遍历缺陷报告中的每一条语句，并调用BEE-tool模型进行分类
    report_label_set = []
    for sentence in sentence_list:
        sent_labels = predict(sentence, model, tokenizer, args)
        report_label_set.extend(sent_labels)

    # 判断缺陷报告中是否含有OB, EB,SR
    if len(set(report_label_set)) == 3:
        return True
    else:
        return False


def predict_multi_data(args):
    """预测多条缺陷报告样本数据"""
    model, tokenizer = create_bert_model(args)
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    for project in project_list:
        perfect_num = 0
        args.bert_project_result_path = os.path.join(args.bert_result_path, project)
        # 创建项目文件保存路径
        if not os.path.exists(args.bert_project_result_path):
            os.mkdir(args.bert_project_result_path)
        if not os.path.exists(os.path.join(args.bert_project_result_path, 'perfect_data')):
            os.mkdir(os.path.join(args.bert_project_result_path, 'perfect_data'))

        # 遍历每个项目的缺陷报告
        filelist = os.listdir(os.path.join(args.json_bug_path, project))
        p_bar = tqdm(filelist, desc="iter", total=len(filelist))
        for idx, filename in enumerate(p_bar):
            p_bar.set_description(f"{project}: No.{idx}")

            # load json data
            if filename.split(".")[-1] == 'json':
                with open(os.path.join(args.json_bug_path, project, filename), 'r', encoding='utf-8') as f:
                    report = json.load(f)
                f.close()
                # 使用BEE-tool对缺陷报告语句进行分类，格式化文件
                perfect_num += predict_and_format(report, model, tokenizer, args)

        print("==={} has {} perfect sample===".format(project, perfect_num))
    
    # Close Stanford NLP
    if stanford_nlp:
        close_nlp(stanford_nlp)


if __name__ == '__main__':
    # 解析参数
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    config = bert_parse_arguments()
    # 加载数据集
    # predict_one_data("", config)
    # 预测缺陷定位项目的所有项目的缺陷报告
    # predict_multi_data(config)
