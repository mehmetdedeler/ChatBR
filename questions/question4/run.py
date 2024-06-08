import argparse
import json
import os

from tqdm import tqdm

from questions.question2.classifier_predict import is_report_perfect
from questions.question2.evaluate import evaluate_gpt_generate

from eval_sim import get_logger
from questions.question2.run import call_ChatGPT


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


def build_prompt_data(data_path, save_path):
    # 数据特征
    del_list = [['OB'], ['EB'], ['SR'], ['OB', 'EB'], ['OB', 'SR'], ['EB', 'SR']]
    # 打开prompt文件
    with open('./prompt.json', 'r') as f:
        prompt_json = json.load(f)
    f.close()

    for key, prompt in prompt_json.items():
        prompt_save_path = os.path.join(save_path, key)
        if not os.path.exists(prompt_save_path):
            os.mkdir(prompt_save_path)
        for project in project_list:
            project_data_path = os.path.join(data_path, project, 'selected_data')
            project_save_path = os.path.join(prompt_save_path, project)

            if not os.path.exists(project_save_path):
                os.mkdir(project_save_path)

            # 遍历选中的样本进行扩充整理
            file_list = os.listdir(project_data_path)
            for filename in file_list:
                # 读取选中的缺陷报告样本
                with open(os.path.join(project_data_path, filename), 'r') as f:
                    report_json = json.load(f)
                f.close()

                # 生成并保存不同版本的 bug_report
                for del_keys in del_list:
                    modified_report = report_json.copy()
                    for key in del_keys:
                        if key in modified_report:
                            modified_report[key] = ""
                    # 保存为 txt 文件
                    with open(
                            f"{os.path.join(project_save_path, filename.split('.')[0])}_remove_{'_'.join(del_keys)}.txt",
                            'w') as f:
                        f.write(prompt + "\n<BUG REPORT>\n" + str(modified_report) + "\n</BUG REPORT>")
                    f.close()


def load_sample_call_llm(args):
    # 遍历每个项目
    for prompt_idx in range(3):
        for project in project_list:
            project_data_path = os.path.join(args.llm_data_path, f'prompt_{prompt_idx}', project)
            gen_data_path = os.path.join(args.gen_data_path, f'prompt_{prompt_idx}', project)

            # 遍历每个项目的缺陷报告文件
            filelist = os.listdir(project_data_path)
            p_bar = tqdm(filelist, total=len(filelist), desc='file index')
            for idx, filename in enumerate(p_bar):
                p_bar.set_description(f"{project}: file No.{idx}")
                # 读取缺陷报告文件
                name, ext = os.path.splitext(filename)
                if ext == 'txt':
                    with open(os.path.join(project_data_path, filename), 'r') as f:
                        bug_report = f.read()
                    # 调用ChatGPT接口，得到返回值
                    response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
                    # 判断返回结果是否正确
                    call_cnt = 0
                    while call_cnt < 5:
                        if response is not None:
                            try:
                                with open(os.path.join(gen_data_path, f"{name}_gpt_{call_cnt}.json"), 'w') as f:
                                    json.dump(response, f)
                            except:
                                print('====save response error====')
                                continue
                            if check_new_report(response['choices'][0]['message']['content']):
                                break
                        else:
                            response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
                        call_cnt += 1
                else:
                    print("非txt文件")
                    continue


def check_new_report(response):
    """检查新生成的report是否符合格式"""
    try:
        # 检查是否是json格式
        report_dict = json.loads(response, strict=False)
        # 检查OB, EB ,SR不为空且含有OB, EB, SR标签
        if report_dict['OB'].strip() != "" and report_dict['EB'].strip() != "" and report_dict['SR'].strip() != "" \
                and is_report_perfect(response['choices'][0]['message']['content'], args):
            return True
        else:
            return False
    except:
        return False


def transfer_data_type():
    """将GPT返回格式转换成json格式"""
    for idx in range(0, 3):
        for project in project_list:
            cnt = 0
            project_path = os.path.join(f'./generate_data/prompt_{idx}/{project}')
            save_path = os.path.join(f'./cleaned_data/prompt_{idx}/{project}')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            filelist = os.listdir(project_path)
            for file in filelist:
                try:
                    with open(os.path.join(project_path, file), 'r', encoding='utf-8') as f:
                        open_rst_dict = json.load(f)
                    f.close()
                    report_text = open_rst_dict['choices'][0]['message']['content']
                    report_dict = json.loads(report_text, strict=False)
                    with open(os.path.join(save_path, file), 'w', encoding='utf-8') as f:
                        json.dump(report_dict, f)
                    f.close()
                except:
                    cnt += 1
                    logger.exception(f"prompt index: {idx}, project: {project}, filename: {file}")
            logger.info(f"prompt index: {idx}, project: {project}, cnt: {cnt}")


if __name__ == '__main__':
    # 项目列表
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    logger = get_logger()
    args = parse_run_arguments()

    # 1. build dataset
    build_prompt_data('../question2/predict_data/bert_new/', './prompt_data')

    # 2. 加载数据集，调用ChatGPT
    load_sample_call_llm(args)

    # 3.解析GPT结果
    transfer_data_type()

    # 4.评估GPT生成结果
    evaluate_gpt_generate("../question4/cleaned_data/prompt_0")
