import argparse
import json
import os
import pickle
import re

from tqdm import tqdm

from javaUtils.data_model import Project


def parse_Arguments():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_data_path', default='D:/projects/pythonProjects/locate_dataset/pickles/',
                        help='原始pkl文件路径')
    parser.add_argument('--json_bug_path',
                        default='D:/projects/pythonProjects/re_report/questions/question2/origin_data/',
                        help='pkl保存文件路径')
    return parser.parse_args()


def parse_pkl_bug_report(args, filename):
    """将缺陷定位项目的缺陷报告转换为json格式"""
    with open(os.path.join(args.pkl_data_path, filename), 'rb') as f:
        raw_data: Project = pickle.load(f)
    # 创建目标文件夹
    dist_path = os.path.join(args.json_bug_path, filename.split('_')[0])
    os.makedirs(dist_path, exist_ok=True)
    # 遍历原始数据中的缺陷，提取缺陷的id，标题和描述
    for bug in tqdm(raw_data.bugs.values(), desc="iter"):
        bug_item = {
            "bug_id": bug.bug_id,
            "title": re.sub(r'\s{2,}|[*#\->_\n\t]+|\[\s+\]|&nbsp;', ' ', bug.bug_summary).strip(),
            "description": re.sub(r'\s{2,}|[*#\->_\n\t]+|\[\s+\]|&nbsp;', ' ', bug.bug_description).strip()
        }
        with open(os.path.join(dist_path, bug.bug_id + '.json'), 'w') as f:
            json.dump(bug_item, f)


if __name__ == '__main__':
    args = parse_Arguments()
    # 将pkl格式的缺陷定位数据集转换为json格式
    filenames = os.listdir(args.pkl_data_path)
    for filename in filenames:
        if filename.endswith('_0.pkl'):
            parse_pkl_bug_report(args, filename)
