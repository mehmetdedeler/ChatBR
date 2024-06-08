import argparse
import json
import os

from tqdm import tqdm

from questions.gpt_utils import call_ChatGPT
from questions.log_utils import get_logger
from questions.question2.classifier_predict import is_report_perfect, create_bert_model


def parse_run_arguments():
    """解析参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--hidden_dropout_prob', default=0.3)
    parser.add_argument('--max_length', default=128)
    parser.add_argument('--num_labels', default=3)
    parser.add_argument('--model_path', default='../question1/model/bert-tuning-01')
    parser.add_argument('--llm_data_path', default='./prompt_data')
    # parser.add_argument('--gen_data_path', default='./generate_data')
    parser.add_argument('--length_limit', default=2000)
    arguments = parser.parse_args()

    return arguments


def open_prompt_file(file_path):
    with open(file_path, 'r') as f:
        file_content = f.read()
    file_content = file_content.encode('utf-8', 'ignore').decode('utf-8')
    return file_content


def open_json_file(file_path):
    with open(file_path, 'r') as f:
        file_content = json.load(f)
    return file_content


def save_file(file_path, response):
    with open(file_path, 'w') as f:
        json.dump(response, f)


def new_empty_response():
    return {
        "id": "",
        "description": "",
        "OB": "",
        "EB": "",
        "SR": ""
    }


def build_prompt_data(data_path='../question2/predict_data/bert_new/', save_path='./prompt_data'):
    del_list = [['OB'], ['EB'], ['SR'], ['OB', 'EB'], ['OB', 'SR'], ['EB', 'SR']]
    with open('./prompt.json', 'r') as f:
        prompt_json = json.load(f)

    for key, prompt in prompt_json.items():
        prompt_save_path = os.path.join(save_path, key)
        if not os.path.exists(prompt_save_path):
            os.mkdir(prompt_save_path)
        for project in project_list:
            # perfect sample path
            project_data_path = os.path.join(data_path, project, 'selected_data')
            # gpt prompt sample path
            project_save_path = os.path.join(prompt_save_path, project)
            if not os.path.exists(project_save_path):
                os.mkdir(project_save_path)

            # 遍历前一半样本进行扩充整理
            file_list = os.listdir(project_data_path)
            for filename in file_list[:len(file_list) // 2]:
                with open(os.path.join(project_data_path, filename), 'r') as f:
                    report_json = json.load(f)

                # 生成并保存不同版本的 bug_report
                for del_keys in del_list:
                    modified_report = report_json.copy()
                    for key in del_keys:
                        if key in modified_report:
                            modified_report[key] = ""
                    name, ext = os.path.splitext(filename)
                    with open(f"{os.path.join(project_save_path, name)}_remove_{'_'.join(del_keys)}.txt", 'w') as f:
                        f.write(prompt + "\n<BUG REPORT>\n" + str(modified_report) + "\n</BUG REPORT>")


def load_sample_call_llm(args):
    """调用chatgpt"""
    for prompt_idx in range(2, 3):  # prompt索引
        for project in project_list:  # 项目名称
            project_data_path = os.path.join(args.llm_data_path, f'prompt_{prompt_idx}', project)
            gen_data_path = os.path.join(args.gen_data_path, f'prompt_{prompt_idx}', project)
            prompt_filelist = os.listdir(project_data_path)

            if not os.path.exists(gen_data_path):
                os.makedirs(gen_data_path, exist_ok=True)
            else:
                # 找到最后一个生成文件的文件名, 调用ChatGPT的次数
                gen_filelist = os.listdir(gen_data_path)
                if len(gen_filelist):
                    gen_final_prompt_prefix = gen_filelist[-1].split("_gpt_")[0]
                    gen_final_prompt_filename = gen_final_prompt_prefix + ".txt"
                    gen_final_subfix_cnt = int(gen_filelist[-1].split("_gpt_")[1].strip(".json"))
                    # 判断最后一个文件是否是高质量报告
                    gen_final_report = open_json_file(os.path.join(gen_data_path, gen_filelist[-1]))
                    if (gen_final_subfix_cnt < 4) and (not check_new_report(gen_final_report, args)):
                        # 记录调用ChatGPT次数，当调用大于5次，跳过该文件
                        bug_prompt = open_prompt_file(os.path.join(project_data_path, gen_final_prompt_filename))
                        call_cnt = gen_final_subfix_cnt + 1
                        while call_cnt < 5:
                            flag, response = call_ChatGPT(bug_prompt, model="gpt-3.5-turbo")
                            if flag == 0:  # 成功调用
                                try:
                                    response = json.loads(response)
                                except:
                                    logger.info(f'<====load response to json error====>')
                                    response = new_empty_response()
                                finally:
                                    save_file(os.path.join(gen_data_path,
                                                           f"{gen_final_prompt_prefix}_gpt_{call_cnt}.json"), response)
                                    if check_new_report(response, args):
                                        break
                                    call_cnt += 1
                    # 找到该文件的prompt文件在目录中的位置, 获取之后的所有prompt文件
                    file_index = prompt_filelist.index(
                        os.path.basename(os.path.join(project_data_path, gen_final_prompt_filename)))
                    prompt_filelist = prompt_filelist[file_index + 1:]

            # 遍历prompt文件生成新文件
            p_bar = tqdm(prompt_filelist, total=len(prompt_filelist), desc='file')
            for idx, file in enumerate(p_bar):
                p_bar.set_description(f"{project}, No.{idx}")
                # 遍历.txt后缀的prompt文件
                filename, ext = os.path.splitext(file)
                if ext == '.txt':
                    # 打开prompt文件, 记录调用ChatGPT次数，当调用大于5次，跳过该文件
                    bug_report = open_prompt_file(os.path.join(project_data_path, file))
                    call_cnt = 0
                    while call_cnt < 5:
                        flag, response = call_ChatGPT(bug_report, model="gpt-3.5-turbo")
                        if flag == 0:  # 成功调用
                            try:
                                response = json.loads(response)
                            except:
                                logger.info(f'<====load response to json error====>')
                                response = new_empty_response()
                            finally:
                                save_file(os.path.join(gen_data_path, f"{filename}_gpt_{call_cnt}.json"), response)
                                if check_new_report(response, args):
                                    break
                                call_cnt += 1
                else:
                    logger.info("====no txt file====")
                    continue


def check_new_report(response, args):
    """检查新生成的report是否符合格式"""
    try:
        response_text = ' '.join(response.values())
        # 检查OB, EB ,SR不为空且含有OB, EB, SR标签
        if len(response['OB'].strip()) > 5 and len(response['EB'].strip()) > 5 and len(response['SR'].strip()) > 5 \
                and is_report_perfect(response_text, args):
            return True
        else:
            logger.info('<====generated report is no perfect report====>')
            return False
    except:
        logger.exception(f"<====generated report format error====>")
        return False


def transfer_data_type():
    """将GPT返回格式转换成json格式"""
    for idx in range(3):
        for project in project_list:
            project_base_path = os.path.join(f"./prompt_data/prompt_{idx}", project)  # prompt file
            project_path = os.path.join(f'round1/generate_data/prompt_{idx}', project)  # gpt generate file
            project_save_path = os.path.join(f'./cleaned_data/prompt_{idx}', project)  # final result file
            if not os.path.exists(project_save_path):
                os.makedirs(project_save_path)
            prompt_filelist = os.listdir(project_base_path)
            gpt_filelist = os.listdir(project_path)
            for file in prompt_filelist:
                filename, ext = os.path.splitext(file)
                try:
                    # gpt final response
                    final_file = [file for file in gpt_filelist if file.startswith(f"{filename}_gpt_")][-1]
                    with open(os.path.join(project_path, final_file), 'r', encoding='utf-8') as f:
                        gpt_rst_dict = json.load(f)
                    final_report_dict = json.loads(gpt_rst_dict['choices'][0]['message']['content'], strict=False)
                    with open(os.path.join(project_save_path, final_file), 'w', encoding='utf-8') as f:
                        json.dump(final_report_dict, f)
                    f.close()
                except:
                    logger.exception(f"===prompt: {idx}, project: {project}, filename: {filename} format error===")
                    continue


if __name__ == '__main__':
    # 项目列表
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    logger = get_logger()
    args = parse_run_arguments()
    args.model, args.tokenizer = create_bert_model(args)

    # # 1. build dataset
    # build_prompt_data('../question2/predict_data/bert_new/', './prompt_data')

    # # 2. 加载数据集，调用ChatGPT
    # for round_cnt in range(2, 6):
    round_cnt = 1
    args.gen_data_path = f"./round{round_cnt}/generate_data"
    load_sample_call_llm(args)
