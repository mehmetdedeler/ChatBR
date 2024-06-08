import argparse
import os
import json
import pandas as pd
import gensim
import torch
from nltk.tokenize import word_tokenize
from scipy import spatial
from torch.autograd.grad_mode import F
from transformers import AutoTokenizer, AutoModel

from questions.log_utils import get_logger

# 定义一些常量和参数
project_list = ["SWT"]
feature_list = ['OB', 'EB', 'SR', 'OB_EB', 'OB_SR', "EB_SR"]


def parse_arguments():
    """解析参数"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--bert_max_length', default=512)
    parser.add_argument('--bert_model_path', default="../question1/model/bert-base")
    parser.add_argument('--project_data_path', default="../question2/predict_data/bert_new", type=str)
    parser.add_argument('--embedding_path', default="../../utils/embed_model/GoogleNews-vectors-negative300.bin")

    parser.add_argument('--prompt_data_path', default="../question7/prompt_data/prompt_0")

    arguments = parser.parse_args()

    return arguments


# 定义一个函数，用于打开json文件并返回数据
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def parse_base_data(project, prompt_data_path):
    filelist = os.listdir(os.path.join(prompt_data_path, project))
    filelist = [file.split("_")[0] for file in filelist]
    return set(filelist)


# 定义一个函数，用于计算两个文本的余弦相似度
def cosine_similarity_use_word2vec(text1, text2, model):
    score = 0
    try:
        score = 1 - spatial.distance.cosine(
            model.get_mean_vector(word_tokenize(text1)),
            model.get_mean_vector(word_tokenize(text2))
        )
    except RuntimeWarning as e:
        logger.info(e)
    return score


# 定义一个函数，用于计算两个文本的余弦相似度
def cosine_similarity_use_bert(text1, text2, tokenizer, model, device, bert_max_length):
    inputs = tokenizer(text1, text2, return_tensors="pt", padding=True, truncation=True,
                       max_length=bert_max_length)
    inputs = inputs.to(device)

    outputs = model(**inputs)

    embeddings1 = outputs.last_hidden_state[0][0].unsqueeze(0)
    embeddings2 = outputs.last_hidden_state[0][1].unsqueeze(0)
    return F.cosine_similarity(embeddings1, embeddings2).item()


def process_project(project, model, tokenizer=None, device=None, use_bert=False):
    project_result_list = []
    # base report list, ChatGPT生成的缺陷报告列表
    base_filelist = parse_base_data(project, args.prompt_data_path)
    gen_filelist = os.listdir(os.path.join(args.gen_data_path, project))

    # ChatGPT生成的缺陷报告列表
    for base_file in base_filelist:
        base_report = load_json_file(
            os.path.join(args.project_data_path, project, 'selected_data', f"{base_file}.json"))

        # 遍历GhatGPT生成的特征组合的缺陷报告
        for feature_item in feature_list:
            try:
                gen_file = [f for f in gen_filelist if f.startswith(f"{base_file}_remove_{feature_item}_gpt")][-1]
                gpt_bug_report = load_json_file(os.path.join(args.gen_data_path, project, gen_file))
            except Exception as e:
                logger.info(f"<==project: {project}, filename: {base_file}_{feature_list}, {e}==>")
                continue

            # 遍历特征组合中的每一个特征
            feature_names = feature_item.split('_')
            for feature in feature_names:
                # 计算原始缺陷报告和GPT生成的特征的语义相似度
                similarity = 0
                try:
                    if use_bert:
                        similarity = cosine_similarity_use_bert(base_report[feature], gpt_bug_report[feature],
                                                                tokenizer, model, device, args.bert_max_length)
                    else:
                        similarity = 1 - spatial.distance.cosine(
                            model.get_mean_vector(word_tokenize(base_report[feature])),
                            model.get_mean_vector(word_tokenize(gpt_bug_report[feature]))
                        )
                except Exception as e:
                    # if int(gen_file.split("_gpt_")[-1].strip(".json")) == 4:
                    #     logger.info(
                    #         f"<====round: {args.round_idx}, prompt: {args.prompt_idx}, {project}, {gen_file}, {e}====>"
                    #     )
                    # else:
                    #     logger.info(
                    #         f"<<**** round: {args.round_idx}, prompt: {args.prompt_idx}, {project}, {gen_file} ****>>"
                    #     )
                    pass
                finally:
                    project_result_list.append([base_file, '_'.join(feature_names), feature, similarity])
                    # logger.info(f"Project: {project}, Bug id: {base_file}, Features: {' '.join(feature_names)}, "
                    #             f"Feature: {feature}, Similarity score: {similarity:.3f}")
    return project_result_list


def calculate_sim_use_word2vec(args):
    # 加载词向量模型
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(args.embedding_path, binary=True)

    # 遍历每个项目
    for project in project_list:
        # 处理项目数据并保存结果
        project_result_list = process_project(project, word2vec_model)
        df = pd.DataFrame(project_result_list, columns=['bug_id', 'feature_combine', 'feature_name', 'score'])
        if not os.path.exists(args.similarity_data_path):
            os.makedirs(args.similarity_data_path)
        df.to_csv(os.path.join(args.similarity_data_path, f'{project}_similarity_result_word2vec.csv'))


def calculate_sim_use_bert():
    # 加载词向量模型
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_path)
    model = AutoModel.from_pretrained(args.bert_model_path)
    # 将模型移动到合适的设备上（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 遍历每个项目
    for project in project_list:
        # 处理项目数据并保存结果
        project_result_list = process_project(project, tokenizer, model, device)
        df = pd.DataFrame(project_result_list, columns=['bug_id', 'feature_combine', 'feature_name', 'score'])
        df.to_csv(os.path.join(args.similarity_data_path, f'{project}_similarity_result_bert.csv'))


def calculate_feature_score(args):
    """计算每一个项目的每一个特征的平均语义相似度"""

    # 定义一个函数，用于计算特征组合中的特征的平均值
    def calculate_mean(dataframe, feature_com):
        # 选取特征组合对应的数据
        dataframe = dataframe[dataframe['feature_combine'].isin(feature_com)]
        # 按照特征分组，计算平均值
        mean = dataframe.groupby('feature_name')['score'].mean()
        return mean

    def calculate_single_mean(dataframe, feature_com):
        # 计算缺失两个特征的所有bug报告中OB, EB, SR单个特征的平均语义
        dataframe = dataframe[dataframe['feature_combine'] == feature_com]
        mean = dataframe.groupby('feature_name')['score'].mean()
        return mean

    for project in project_list:
        # 读取项目的相似度数据
        df = pd.read_csv(os.path.join(args.similarity_data_path, f'{project}_similarity_result_word2vec.csv'))

        # 计算仅缺失一个元素的平均值
        miss_one_values = ['OB', 'EB', 'SR']
        rst_one_feature = calculate_mean(df, miss_one_values)

        # 计算所有缺失两个元素中每一个元素的平均语义相似度
        miss_two_values = ['OB_EB', 'OB_SR', 'EB_SR']
        rst_two_feature = calculate_mean(df, miss_two_values)

        # 计算缺失单个两个元素的每一个元素的平均语义
        miss_two_feature_mean_df = pd.DataFrame()
        for miss_two_values_item in miss_two_values:
            miss_two_feature_mean_df = pd.concat([miss_two_feature_mean_df,
                                                  calculate_single_mean(df, miss_two_values_item)])

        # 计算所有OB, EB, SR的平均值
        rst_all_combine = calculate_mean(df, feature_list)

        feature_mean_df = pd.concat([rst_one_feature, rst_two_feature, rst_all_combine], axis=1)
        feature_mean_df.columns = ['miss_one', 'miss_two', 'all_avg']
        # 输出结果
        miss_one_list = feature_mean_df.iloc[:, 0].tolist()
        miss_two_list = miss_two_feature_mean_df.iloc[:, 0].tolist()
        logger.info(f"\n{project}:")
        for i in miss_one_list:
            logger.info(i)
        for i in miss_two_list:
            logger.info(i)


if __name__ == '__main__':
    logger = get_logger()
    args = parse_arguments()
    for round_idx in range(1, 6):
        for prompt_idx in range(3):
            args.gen_data_base_path = f"../question7/round{round_idx}/"
            args.gen_data_path = f"{args.gen_data_base_path}/generate_data/prompt_{prompt_idx}"
            args.similarity_data_path = f"{args.gen_data_base_path}/similarity_result/prompt_{prompt_idx}"
            args.round_idx = round_idx
            args.prompt_idx = prompt_idx
            calculate_sim_use_word2vec(args)
            calculate_feature_score(args)
            logger.info(f"<=====round: {round_idx}, prompt: {prompt_idx} finished!=====>")
