import json
import os

import gensim
import pandas as pd
from gensim.models import Word2Vec
from nltk import word_tokenize
from scipy import spatial
from transformers import AutoTokenizer, AutoModel
import torch

from classifier_predict import bert_parse_arguments

project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
feature_list = ['OB', 'EB', 'SR', 'OB_EB', 'OB_SR', 'EB_SR']


def use_bert_cal_sim(data_path='./generate_data'):
    args = bert_parse_arguments()
    args.model_path = '../question1/model/bert-base'
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)

    # 将模型移动到合适的设备上（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    args.device = device

    for project in project_list:
        project_result_list = []
        # 含有OB, EB, SR的缺陷报告路径
        project_path = os.path.join('./predict_data/bert', project, 'selected_data')
        # ChatGPT生成的缺陷报告列表
        gen_report_list = os.listdir(os.path.join(data_path, project))
        # 遍历含有OB, EB, SR的缺陷报告列表
        for filename in os.listdir(project_path):
            # 获取bug_id即文件名
            bug_id = filename.split('.')[0]
            # 打开原始缺陷报告
            with open(os.path.join(project_path, filename), 'r') as f:
                bug_report = json.load(f)
            f.close()

            try:
                # 遍历GhatGPT生成的特征组合的缺陷报告
                for feature_item in feature_list:
                    # 打开特征组合缺陷报告, 需要找到成功生成符合条件的那一个文件
                    gen_filename = [file for file in gen_report_list if
                                    file.startswith(f"{bug_id}_remove_{feature_item}_gpt_")][-1]
                    with open(os.path.join(data_path, project, gen_filename), 'r') as f:
                        gpt_bug_report = json.load(f)
                    f.close()

                    # 遍历每一个特征
                    feature_names = feature_item.split('_')
                    for feature in feature_names:
                        # 计算原始缺陷报告和GPT生成的特征的语义相似度
                        inputs = tokenizer(bug_report[feature], gpt_bug_report[feature], return_tensors="pt",
                                           padding=True, truncation=True, max_length=args.max_length)
                        inputs = inputs.to(args.device)
                        # Pass the tokenized sentences through the BERT model
                        outputs = model(**inputs)
                        # Extract the embeddings for each sentence
                        embeddings1 = outputs.last_hidden_state[0][0].unsqueeze(0)
                        embeddings2 = outputs.last_hidden_state[0][1].unsqueeze(0)
                        # Calculate the cosine similarity between the embeddings
                        cosine_similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()

                        project_result_list.append([bug_id, ' '.join(feature_names), feature, cosine_similarity])
                        print(f"===Project: {project}, Bug id: {bug_id}, Features:{' '.join(feature_names)}, "
                              f"Feature: {feature}, Similarity score: {cosine_similarity:.3f}===")
            except Exception as e:
                print(f"Got an exception! relate information: project: {project}, bug id: {bug_id}")
                raise e
        df = pd.DataFrame(project_result_list, columns=['bug_id', 'feature_com', 'feature', 'score'])
        df.to_csv(f'./similarity_data/{project}_similarity_result_bert.csv')


def use_word2vec_cal_sim(embedding_path, data_path="./generate_data"):
    # model = Word2Vec.load(embedding_path).wv
    model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=True)

    for project in project_list:

        project_result_list = []
        # 被选中的含有OB, EB, SR的缺陷报告
        project_path = os.path.join('./predict_data/bert', project, 'selected_data')
        # ChatGPT生成的缺陷报告列表
        gen_report_list = os.listdir(os.path.join(data_path, project))

        for filename in os.listdir(project_path):
            # 获取bug_id即文件名
            bug_id = filename.split('.')[0]
            # 打开原始缺陷报告
            with open(os.path.join(project_path, filename), 'r') as f:
                bug_report = json.load(f)
            f.close()

            try:
                # 遍历GhatGPT生成的特征组合的缺陷报告
                for feature_item in feature_list:
                    # 打开特征组合缺陷报告, 需要找到成功生成符合条件的那一个文件
                    gen_filename = [file for file in gen_report_list if
                                    file.startswith(f"{bug_id}_remove_{feature_item}_gpt_")][-1]
                    with open(os.path.join(data_path, project, gen_filename), 'r') as f:
                        gpt_bug_report = json.load(f)
                    f.close()

                    # 遍历每一个特征
                    feature_names = feature_item.split('_')
                    for feature in feature_names:
                        # 计算原始缺陷报告和GPT生成的特征的语义相似度
                        cosine_similarity = 1 - spatial.distance.cosine(
                            model.get_mean_vector(word_tokenize(bug_report[feature])),
                            model.get_mean_vector(word_tokenize(gpt_bug_report[feature]))
                        )

                        project_result_list.append([bug_id, ' '.join(feature_names), feature, cosine_similarity])
                        print(f"===Project: {project}, Bug id: {bug_id}, Features:{' '.join(feature_names)}, "
                              f"Feature: {feature}, Similarity score: {cosine_similarity:.3f}===")
            except Exception as e:
                print(f"Got an exception! relate information: project: {project}, bug id: {bug_id}")
                raise e
        # df = pd.DataFrame(project_result_list, columns=['bug_id', 'feature_com', 'feature', 'score'])
        # df.to_csv(f'./similarity_data/{project}_similarity_result.csv')


def calculate_feature_score():
    """计算每一个项目的每一个特征的平均语义相似度"""

    for project in project_list:
        df = pd.read_csv(f'./similarity_data/{project}_similarity_result.csv')

        # 统计所有OB, EB, SR的平均值
        rst_all_combine = df.groupby('feature')['score'].mean()

        # 统计仅缺失一个元素的平均值
        one_excluded_values = ['OB_EB', 'OB_SR', 'EB_SR']
        df_one_feature = df[~df['feature_com'].isin(one_excluded_values)]
        rst_one_feature = df_one_feature.groupby('feature')['score'].mean()

        # 统计两个
        two_excluded_values = ['OB', 'EB', 'SR']
        df_two_feature = df[~df['feature_com'].isin(two_excluded_values)]
        rst_two_feature = df_two_feature.groupby('feature')['score'].mean()

        # 统计缺失两个元素的缺陷报告和缺失一个元素的缺陷报告的相似度是否有差别
        print(f"==={project}\nmiss_one_feature: {rst_one_feature}, miss_two_feature: {rst_two_feature},"
              f"miss_non_feature: {rst_all_combine}===")


def calculate_feature_in_two():
    """计算每种特征在特征组合中的语义相似度"""
    combine_list = ['OB EB', 'OB SR', 'EB SR']
    for combine_item in combine_list:
        for project in project_list:
            df = pd.read_csv(f'./similarity_data/{project}_similarity_result.csv')
            # 选取feature_com列的元素值为OB_EB的所有行
            df = df[df['feature_com'] == combine_item]

            # 统计feature列中元素为OB和EB的元素值的平均值
            mean = df.groupby('feature')['score'].mean()
            print(project, mean.index.tolist())
            print(mean.iloc[0], mean.iloc[1])


if __name__ == '__main__':
    # use_bert_cal_sim()
    use_word2vec_cal_sim('../../utils/embed_model/GoogleNews-vectors-negative300.bin')

    # calculate_feature_score()
    # calculate_sentences()
    # calculate_feature_in_two()
