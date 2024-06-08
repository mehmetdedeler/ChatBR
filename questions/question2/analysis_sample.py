"""collect prefect dataset after bee-tool predict and analysis dataset info"""
import json
import os
from shutil import copyfile
import pandas as pd


def analysis_llm_data(json_data_path):
    """analysis perfect sample info"""
    for project in project_list:
        bug_report_len_list = []
        project_perfect_data_path = os.path.join(json_data_path, project, 'perfect_data')
        csv_file_path = os.path.join(json_data_path, '{}_len_info.csv'.format(project))

        file_list = os.listdir(os.path.join(project_perfect_data_path))
        for idx, filename in enumerate(file_list):
            if filename.split('.')[-1] == 'json':
                with open(os.path.join(project_perfect_data_path, filename), 'r') as f:
                    bug_report = json.load(f)
                f.close()

                desc_len = len(bug_report['description'] + bug_report['OB'] + bug_report['EB'] +
                               bug_report['SR'])
                bug_report_len_list.append([filename.split('.')[0], len(bug_report['title']), desc_len,
                                            len(bug_report['title'])+desc_len])

        df = pd.DataFrame(data=bug_report_len_list, columns=['bug_id', 'title', 'desc', 'total'])
        df.to_csv(csv_file_path)


def select_dataset_for_llm(json_data_path, llm_data_path, K=2000):
    """分析BERT数据集"""
    # 数据特征
    del_list = [['OB'], ['EB'], ['SR'], ['OB', 'EB'], ['OB', 'SR'], ['EB', 'SR']]
    for project in project_list:
        project_data_path = os.path.join(json_data_path, project, 'perfect_data')  # 项目的高质量缺陷报告路径
        llm_project_path = os.path.join(llm_data_path, project)  # 项目的llm文件存放路径
        csv_file_path = os.path.join(json_data_path, "{}_len_info.csv".format(project))
        if not os.path.exists(llm_project_path):
            os.mkdir(llm_project_path)

        # 筛选项目中缺陷报告长度小于K的样本
        len_df = pd.read_csv(csv_file_path)
        selected_list = len_df[len_df['desc'] < K]['bug_id'].tolist()
        for bug_id in selected_list:
            # 读取选中的缺陷报告样本
            with open(os.path.join(project_data_path, "{}.json".format(bug_id)), 'r') as f:
                report_json = json.load(f)
            f.close()
            # 生成并保存不同版本的 bug_report
            for i, del_keys in enumerate(del_list):
                modified_report = report_json.copy()
                for key in del_keys:
                    if key in modified_report.keys():
                        modified_report[key] = ""
                try:
                    with open(f"{os.path.join(llm_project_path, bug_id)}_remove_{'_'.join(del_keys)}.txt", 'w') as file:
                        with open("../question4/prompt.json", 'r') as prompt_file:
                            prompt_json = json.load(prompt_file)
                        prompt_file.close()
                        file.write(prompt_json['prompt_0'] + "\n\n<BUG REPORT>\n" +
                                   str(modified_report) + "\n</BUG REPORT>")
                except Exception as e:
                    print(e)
                finally:
                    file.close()


if __name__ == '__main__':
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    analysis_llm_data("./predict_data/bert_new")
    select_dataset_for_llm('./predict_data/bert_new', './llm_dataset')