import argparse
import json
import os

from tqdm import tqdm

from classifier_predict import predict_multi_data, is_report_perfect
from analysis_sample import analysis_llm_data, select_dataset_for_llm
import sys
sys.path.append('..')
from gpt_utils import call_ChatGPT
try:
    from pkl2json import parse_pkl_bug_report
    from plot_utils import plot_llm_dataset_ring
except ImportError:
    # Create dummy functions if modules don't exist
    def parse_pkl_bug_report(args, filename):
        print(f"Warning: pkl2json module not found, skipping {filename}")
    
    def plot_llm_dataset_ring(result_path, max_length, project_list):
        print(f"Warning: plot_utils module not found, skipping plotting")


# 1. 转换数据类型
# 2. 预测bug报告质量，并将高质量报告格式化为gpt需要的格式, 选择符合长度的bug报告，绘图
# 3. 调用GPT
# 4. 分析实验结果， 绘图

def parse_run_arguments():
    """解析参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_data_path', default='D:/projects/pythonProjects/locate_dataset/pickles/',
                        help='原始pkl文件路径')
    parser.add_argument('--json_bug_path',
                        default='D:/projects/pythonProjects/re_report/questions/question2/origin_data/',
                        help='pkl保存文件路径')
    parser.add_argument('--bert_result_path', default='./predict_data/bert_new/', help='bert模型预测结果路径')
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--num_labels', default=3)
    parser.add_argument('--max_length', default=128)
    parser.add_argument('--hidden_dropout_prob', default=0.3)
    parser.add_argument('--model_path', default='../question1/model/bert-tuning-01')
    parser.add_argument('--llm_data_path', default='./llm_dataset')
    parser.add_argument('--report_max_length', default=2000)

    arguments = parser.parse_args()

    return arguments


def transfer_datatype(args):
    """转换数据类型"""
    filenames = os.listdir(args.pkl_data_path)
    for filename in filenames:
        if filename.endswith('_0.pkl'):
            parse_pkl_bug_report(args, filename)


def run_bert_predict(args):
    # 预测缺陷定位项目的所有项目的缺陷报告
    predict_multi_data(args)
    # 分析经过bert预测后，重构的高质量缺陷报告长度信息
    analysis_llm_data(args.bert_result_path)
    # 选择字符长度小于K的缺陷报告, 将选中的缺陷报告格式化为llm需要的格式
    select_dataset_for_llm(args.bert_result_path, args.llm_data_path, args.report_max_length)
    # plot
    plot_llm_dataset_ring(args.bert_result_path, args.report_max_length, project_list)


def load_sample_call_llm(args):
    # 遍历每个项目
    args.gen_data_path = './generate_data'  # Add missing variable
    if not os.path.exists(args.gen_data_path):
        os.mkdir(args.gen_data_path)

    for project in project_list:
        # 获取文件列表， 遍历文件
        project_llm_data_path = os.path.join(args.llm_data_path, project)
        project_gen_data_path = os.path.join(args.gen_data_path, project)
        if not os.path.exists(project_gen_data_path):
            os.mkdir(project_gen_data_path)

        # 遍历每个项目的缺陷报告文件
        filelist = os.listdir(project_llm_data_path)
        p_bar = tqdm(filelist, total=len(filelist), desc='iter')
        for idx, filename in enumerate(p_bar):
            p_bar.set_description(f"{project}: No.{idx}")
            # 读取缺陷报告文件
            ext = os.path.splitext(filename)[1]
            if ext == '.txt':
                with open(os.path.join(project_llm_data_path, filename), 'r') as f:
                    bug_report = f.read()
                response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
                for i in range(5):
                    if response is not None:
                        with open(os.path.join(project_gen_data_path,
                                               os.path.join(filename.split('.')[0], f"_gpt_{i}.json")), 'w') as f:
                            json.dump(response, f)
                        if is_report_perfect(response['choices'][0]['message']['content'], args):
                            break
                    response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")


if __name__ == '__main__':
    # 解析参数
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    args = parse_run_arguments()
    # 加载数据集，调用ChatGPT
    transfer_datatype(args)
    run_bert_predict(args)
    load_sample_call_llm(args)
