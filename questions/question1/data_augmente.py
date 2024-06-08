import random

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from tqdm import tqdm


def random_insertion(sentence: list, n: int, token_num: int):
    """随机同义词插入"""
    new_sentences = []
    # 产生n个增强样本
    for _ in range(n):
        sentence_copy = sentence.copy()
        # 随机选择1-0.01*len(token)个字符修改
        random_token_num = random.randint(1, token_num)
        selected_words = random.sample(sentence, random_token_num)
        # word = random.choice(sentence)  # 随机选择1个
        for word in selected_words:
            # 获取该单词的同义词列表
            synonyms = wn.synsets(word)
            if synonyms:
                # 如果有同义词，就随机选择一个
                synonym = random.choice(synonyms).name().split('.')[0]
                # 随机插入
                random_idx = random.randint(0, len(sentence_copy) - 1)
                sentence_copy.insert(random_idx, synonym)
        # 保存增强后的样本
        new_sentences.append(' '.join(sentence_copy))

    return new_sentences


def synonym_replacement(sentence: list, n: int, token_num: int):
    """同义词替换"""
    new_sentences = []
    # 产生n个增强样本
    for _ in range(n):
        sentence_copy = sentence.copy()
        # 随机选择1-0.01*len(token)个字符修改
        random_token_num = random.randint(1, token_num)
        selected_words = random.sample(sentence, random_token_num)
        # word = random.choice(sentence)
        for word in selected_words:
            # 获取该单词的同义词列表
            synonyms = wn.synsets(word)
            if synonyms:
                # 如果有同义词，就随机选择一个
                synonym = random.choice(synonyms).name().split('.')[0]
                # 随机插入
                sentence_copy = [synonym if w == word else w for w in sentence_copy]
        new_sentences.append(' '.join(sentence_copy))

    return new_sentences


def random_swap(sentence: list, n: int):
    """随机交换"""
    new_sentences = []
    for _ in range(n):
        if len(sentence) > 3:
            # 随机选择一个索引，不能是最后一个
            index_before, index_after = random.randint(0, len(sentence) - 2), random.randint(0, len(sentence) - 2)
            # 交换单词
            sentence[index_before], sentence[index_after] = sentence[index_after], sentence[index_before]
            new_sentences.append(' '.join(sentence))
        else:  # 句子太短，直接复制
            new_sentences.append(' '.join(sentence))
    return new_sentences


def dataAugmentation(sentence: str, n: int):
    """数据增强"""
    augment_sentences = []
    sentence = nltk.word_tokenize(sentence)
    token_num = int(0.01*len(sentence))  # 变动token的个数
    if token_num <= 1:  # 保证区间的正确性
        token_num = 1
    augment_sentences.extend(random_insertion(sentence, n, token_num))
    augment_sentences.extend(synonym_replacement(sentence, n, token_num))
    augment_sentences.extend(random_swap(sentence, n))
    return augment_sentences


def prepare(data_path, dist_path):
    """数据处理"""
    # 读取文件
    df = pd.read_pickle(data_path)
    # 使用 value_counts() 函数统计每个元素值的个数
    value_counts = df['labels'].value_counts()
    # 打印每个类别的元素值和对应的个数
    for label, count in value_counts.items():
        print(f"before label: {label}, Count: {count}")

    # 数据增强
    data_groups = df.groupby(['labels'])
    for label, group in data_groups:
        print('====before aug Group name: {}, data count: {}===='.format(label[0], group.shape[0]))
        augment_sent, n = [], 0
        # 每一条数据需要增强多少次
        if 5000 < group.shape[0] < 10000:
            n = 2
        elif 3000 < group.shape[0] <= 5000:
            n = 3
        elif 1000 < group.shape[0] <= 3000:
            n = 12
        elif 200 < group.shape[0] < 1000:
            n = 80
        elif 0 < group.shape[0] < 200:
            n = 210
        # 如果group数据需要数据增强
        if n != 0:
            # 遍历group中的每一行数据，数据增强
            p_bar = tqdm(group['text'], total=group.shape[0])
            for idx, text in enumerate(p_bar):
                augment_sent.extend(dataAugmentation(text, n))
                p_bar.set_description('label:{}, idx:{}'.format(label, idx))

            print('====after aug Group name: {}, data count: {}===='.format(label[0], len(augment_sent)))

            augment_sent = pd.DataFrame(augment_sent, columns=['text'])
            augment_sent['labels'] = label[0]
            df = pd.concat([df, augment_sent], ignore_index=True)

    # 去除空值和重复值
    df.dropna(how='any', inplace=True)
    df.drop_duplicates(subset=['text'], inplace=True)
    df = df.reset_index(drop=True)

    # 使用 value_counts() 函数统计每个元素值的个数
    value_counts = df['labels'].value_counts()
    # 打印每个类别的元素值和对应的个数
    for label, count in value_counts.items():
        print(f"after label: {label}, Count: {count}")

    # 保存文件
    print("====total data length: {}, raw data: {}, aug data: {}====".format(df.shape[0], 67376, df.shape[0]-67376))
    df.to_pickle(dist_path)


if __name__ == '__main__':
    pkl_data_path = "./all_statement.pkl"
    pkl_dist_path = "./all_statement_aug.pkl"
    prepare(pkl_data_path, pkl_dist_path)
