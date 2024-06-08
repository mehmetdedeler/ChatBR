# 1. 将bee-tool数据集的xml文件转换为pkl文件，提取xml文件中的每一条语句和语句的标签
# 2. 将缺陷定位项目的缺陷报告转换为json格式
# 导入必要的库
import argparse
import os
import re
import xml.etree.ElementTree as ET
import pandas as pd


def parse_Arguments():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_data_path',
                        default="D:/dataset/1_data_nimbus_labeled_with_desc/1_data_nimbus_labeled_with_desc" \
                                "/1_labeled_data_selected_with_desc/",
                        help='原始xml文件路径')
    parser.add_argument('--xml_dist_path',
                        default="D:/dataset/1_data_nimbus_labeled_with_desc/1_data_nimbus_labeled_with_desc" \
                                "/1_labeled_data_selected_with_desc/sent_data.pkl",
                        help='xml保存文件路径')
    return parser.parse_args()


def convert(x):
    # 定义一个函数，将x转换为1，否则转换为0
    if x == 'x':
        return 1
    else:
        return 0


def parse_xml_bug_report(file_path):
    # 读取xml文件
    tree = ET.parse(file_path)
    root = tree.getroot()
    stat_list = []  # 定义一个空的列表，用于存储提取的数据
    # 遍历xml文件中的每个段落和句子
    for parg in root.findall('.//parg'):
        # 获取段落的标签 将标签转换为数字
        parg_ob, parg_eb, parg_sr = convert(parg.get('ob')), convert(parg.get('eb')), convert(parg.get('sr'))
        for st in parg.findall('.//st'):
            # 获取句子的文本和标签 将标签转换为数字
            text = re.sub(r'\u200B|\u200D|\uFEFF|\n|\t', "", str(st.text).strip())
            if len(re.sub(r'\s+', "", text)) < 5:
                continue
            ob, eb, sr = convert(st.get('ob')), convert(st.get('eb')), convert(st.get('sr'))
            # 如果段落的标签为1，则句子的对应标签也为1. 将文本和标签组成一个元组，添加到列表中
            stat_list.append([text, str([max(ob, parg_ob), max(eb, parg_eb), max(sr, parg_sr)])])

    return stat_list


def xml2pkl(args):
    """将bee-tool数据集的xml文件转换为pkl文件，提取xml文件中的每一条语句和语句的标签"""
    data_path, dist_path, data_list = args.xml_data_path, args.xml_dist_path, []
    for folder in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, folder)):
            for filename in os.listdir(os.path.join(data_path, folder)):
                if filename.split('.')[-1] == 'xml':
                    data_list.extend(parse_xml_bug_report(os.path.join(data_path, folder, filename)))

    # 将列表转换为pandas的DataFrame对象
    df = pd.DataFrame(data_list, columns=['text', 'labels'])

    # 初始样本数据分布
    labels_counter = df['labels'].value_counts()
    print("===data labels counter. total length:{}===".format(df.shape[0]))
    print(labels_counter)

    df.dropna(how='any', inplace=True)  # 去除空值和重复值
    df.drop_duplicates(subset=['text'], inplace=True)
    df = df.reset_index(drop=True)  # 重置索引

    # 去除空值和重复值后的数据样本分布
    labels_counter = df['labels'].value_counts()
    print("===drop_null and drop_duplicates data labels counter. total length:{}===".format(df.shape[0]))
    print(labels_counter)

    # # 筛选出 text 列中以 at 开头的元素 堆栈信息。 随机丢弃一半的元素
    # at_text = df[df["text"].str.startswith("at")]
    # at_text = at_text.sample(frac=0.5, random_state=42)
    # df = pd.concat([df, at_text]).drop_duplicates(subset=['text'], keep=False)
    #
    # # 去除一半堆栈信息后的数据分布
    # labels_counter = df['labels'].value_counts()
    # print("===drop_half_stack data labels counter. total length:{}===".format(df.shape[0]))
    # print(labels_counter)

    df.to_pickle(dist_path)  # 存为csv会在训练的时候出错


if __name__ == '__main__':
    args = parse_Arguments()
    xml2pkl(args)
