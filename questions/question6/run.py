import argparse
import json
import os

from tqdm import tqdm

from questions.question2.classifier_predict import is_report_perfect
from questions.gpt_utils import call_ChatGPT


def parse_run_arguments():
    """解析参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--hidden_dropout_prob', default=0.3)
    parser.add_argument('--max_length', default=128)
    parser.add_argument('--num_labels', default=3)
    parser.add_argument('--model_path', default='../question1/model/bert-tuning-01')
    parser.add_argument('--llm_data_path', default='./prompt_data')
    parser.add_argument('--gen_data_path', default='./generate_data')
    parser.add_argument('--length_limit', default=2000)

    arguments = parser.parse_args()

    return arguments


def load_sample_call_llm(args, model):
    # 遍历每个项目
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    for prompt_idx in range(3):
        for project in project_list:
            # 获取文件列表， 遍历文件
            project_data_path = os.path.join(args.llm_data_path, project)
            # llm生成文件保存路径
            if model == "gpt-3.5-turbo":
                gen_data_path = os.path.join(args.gen_data_path, "GPT_3", project)
            else:
                gen_data_path = os.path.join(args.gen_data_path, "GPT_4", project)

            # 遍历每个项目的缺陷报告文件
            filelist = os.listdir(project_data_path)
            p_bar = tqdm(filelist, total=len(filelist), desc='file index')
            for idx, filename in enumerate(p_bar):
                p_bar.set_description(f"{project}: No.{idx}")
                # 读取缺陷报告文件
                if filename.split('.')[1] == 'txt':
                    with open(os.path.join(project_data_path, filename), 'r') as f:
                        bug_report = f.read()
                    f.close()
                    # 调用ChatGPT接口，得到返回值
                    response = call_ChatGPT(bug_report, model=model)
                    # 判断返回结果是否正确
                    call_cnt = 0
                    while call_cnt < 5:
                        if response is not None:
                            with open(os.path.join(gen_data_path, f"{filename.split('.')[0]}_gpt_{call_cnt}.json"),
                                      'w') as f:
                                json.dump(response, f)
                            if is_report_perfect(response['choices'][0]['message']['content'], args):
                                break
                        else:
                            response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
                        call_cnt += 1
                else:
                    print("非txt文件")
                    continue


if __name__ == '__main__':
    # 解析参数
    args = parse_run_arguments()
    # 加载数据集，调用ChatGPT
    load_sample_call_llm(args, model="gpt-3.5-turbo")
    load_sample_call_llm(args, model="gpt-4.0")
