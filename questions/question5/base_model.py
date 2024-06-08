import argparse
import json
import os
import sys
from typing import List

import regex
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, LlamaForCausalLM, GPTJForCausalLM, \
    AutoModelForCausalLM, XGLMForCausalLM, BertTokenizer, BertConfig, BertForSequenceClassification
sys.path.append('../..')
from questions.log_utils import get_logger
from questions.question2.classifier_predict import classify_report_quality

PROMPT_TEMPLATE = dict(
    PROMPT="Your role is a senior software engineer, you are very good at analyzing and writing bug reports."
           "In your bug reports, it is essential to provide clear and informative statements for the following categories:"
           "- **Observed Behavior (OB):** This section should describe the relevant software behavior, actions, output, or results. Avoid vague statements like \"the system does not work.\""
           "- **Expected Behavior (EB):** This part should articulate what the software should or is expected to do, using phrases like \"should...\", \"expect...\", or \"hope...\". Avoid suggestions or recommendations for bug resolution in this section."
           "- **Steps to Reproduce (SR):** Include user actions or operations that can potentially lead to reproducing the issue. Use phrases like \"to reproduce,\" \"steps to reproduce,\" or \"follow these steps.\""
           "It is possible that the bug report may lack sufficient details in the OB, EB, and SR sections. In such cases, your task is to infer the appropriate details based on the context and supplement the bug report to ensure it contains clear and complete OB/EB/SR statements. Also, improve the wording of these statements for clarity where possible."
           "To facilitate this process, please provide your responses in JSON format as follows:\n{\"id\": \"\", \"title\": \"\", \"description\": \"\", \"OB\": \"\", \"EB\": \"\", \"SR\": \"\"}"
           "<BUG REPORT>\n{report}\n</BUG REPORT>",
)

MODEL_DICT = {
    "gpt-j": GPTJForCausalLM,
    "code-t5": T5ForConditionalGeneration,
    'llama2': LlamaForCausalLM,
    "vicuna-7b": AutoModel
}


def parse_run_arguments():
    """解析参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--hidden_dropout_prob', default=0.3)
    parser.add_argument('--max_length', default=128)
    parser.add_argument('--num_labels', default=3)
    parser.add_argument('--cls_model_path', default='../question1/model/bert-tuning-01')
    parser.add_argument('--gen_model_dir', default='/Se-jiwangjie/prompt/models/')
    parser.add_argument('--data_path', default='./prompt_data/')
    parser.add_argument('--save_path', default='./generate_data')
    parser.add_argument('--projects', default=["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"])
    parser.add_argument('--length_limit', default=2000)
    arguments = parser.parse_args()
    return arguments


def load_gen_model(args):
    args.logger.info(f"=======loading models: {args.gen_model_path.split('/')[-1]}======")
    gen_tokenizer = AutoTokenizer.from_pretrained(args.gen_model_path)

    model_name = args.gen_model_path.split('/')[-1]
    if model_name.startswith('gpt'):
        gen_model = AutoModelForCausalLM.from_pretrained(args.gen_model_path, offload_folder="offload",
                                                         offload_state_dict=True, torch_dtype=torch.float16)
    elif model_name.startswith('llama') or model_name.startswith('vicuna'):
        gen_model = LlamaForCausalLM.from_pretrained(args.gen_model_path, offload_state_dict=True,
                                                     offload_folder="offload", torch_dtype=torch.float16, )
    elif model_name.startswith('incoder'):
        gen_model = XGLMForCausalLM.from_pretrained(args.gen_model_path, offload_state_dict=True,
                                                    offload_folder="offload", torch_dtype=torch.float16)
    else:
        gen_model = AutoModel.from_pretrained(args.gen_model_path, offload_state_dict=True,
                                              offload_folder="offload", torch_dtype=torch.float16)
    gen_model.to(args.device)
    gen_model.eval()
    args.logger.info("=======loaded models: {}======".format(args.gen_model_path.split('/')[-1]))
    return gen_model, gen_tokenizer


def load_classifier_model(args):
    tokenizer = BertTokenizer.from_pretrained(args.cls_model_path)
    model_config = BertConfig.from_pretrained(args.cls_model_path, num_labels=args.num_labels,
                                              hidden_dropout_prob=args.hidden_dropout_prob)
    model = BertForSequenceClassification.from_pretrained(args.cls_model_path, config=model_config)
    model.to(args.device)
    return model, tokenizer


def call_model_to_gen(model, tokenizer, prompt, args):
    # try:
    input_ids = tokenizer(prompt, return_tensors='pt').to(args.device)
    # except TypeError:
    # input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
    # try:
    output_ids = model.generate(input_ids['input_ids'],attention_mask=input_ids['attention_mask'], max_length=1024,
                                pad_token_id=tokenizer.pad_token_id)
    # except ValueError:
    # output_ids = model.generate(encoded_input, max_length=1024)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def tokenize_text(args, text):
    tokens = len(args.gen_tokenizer.tokenize(text))
    if 'bert' in args.gen_model_path.lower() and tokens > 512:
        return False
    elif 'llama2' in args.gen_model_path.lower() and tokens > 2048:
        return False
    elif 'gptJ' in args.gen_model_path.lower() and tokens > 1024:
        return False
    elif 'codeT5' in args.gen_model_path.lower() and tokens > 2048:
        return False
    elif 'vicuna' in args.gen_model_path.lower() and tokens > 2048:
        return False
    else:
        return True


def load_prompt_data(args) -> dict:
    """加载prompt数据"""
    prompt_data = dict()
    for project in args.projects:
        data_list = []
        if os.path.exists(os.path.join(args.data_path, project)):  # if already have dataset
            project_path = os.path.join(args.data_path, project)
            for file in os.listdir(project_path):
                with open(os.path.join(project_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                    data_list.append({
                        "bug_id": file.split("_")[0],
                        'features': file.strip('.txt').split('remove_')[1].split('_'),
                        'prompt': f.read()
                    })
            prompt_data[project] = data_list
        else:  # not have dataset
            project_path = os.path.join(args.data_path, project, 'selected_data')
            if os.path.exists(project_path):
                for file in os.listdir(project_path):
                    with open(os.path.join(project_path, file), 'r') as f:
                        report = json.load(f)
                        # if self.tokenize_text(PROMPT_TEMPLATE['PROMPT'].format(report=str(report))):
                        data_list.append({
                            "bug_id": file.split("_")[0],
                            'features': file.strip('.txt').split('remove_')[1],
                            'prompt': PROMPT_TEMPLATE['PROMPT'].format(report=str(report))
                        })
                prompt_data[project] = parse_data(project, data_list, args.save_path)
    args.logger.info("=======loaded dataset======")
    return prompt_data


def parse_data(project: str, data_list: List, save_path: str) -> List[dict]:
    del_list = [['OB'], ['EB'], ['SR'], ['OB', 'EB'], ['OB', 'SR'], ['EB', 'SR']]
    prompt_list = []

    save_path = os.path.join(save_path, project)
    os.makedirs(save_path, exist_ok=True)

    for data_item in data_list:
        filename, report_data = data_item.keys()[0], data_item.values()[0]

        for del_item in del_list:
            modified_data = report_data.copy()
            for key in del_item:
                if key in modified_data:
                    modified_data[key] = ""

            prompt = PROMPT_TEMPLATE['PROMPT'].format(str(modified_data))
            file_path = f"{os.path.join(save_path, filename.split('.')[0])}_remove_{'_'.join(del_item)}.txt"
            try:
                with open(file_path, 'w') as f:
                    f.write(prompt)
                prompt_list.append({
                    'bug_id': filename.split('.')[0],
                    'features': del_item,
                    'prompt': prompt
                })
            except Exception as e:
                raise RuntimeError(e)
    return prompt_list


def check_new_report(cls_model, cls_tokenizer, report_dict, args):
    # 检查OB, EB ,SR不为空且含有OB, EB, SR标签
    try:
        ob = report_dict['OB'].strip()
        eb = report_dict['EB'].strip()
        sr = report_dict['SR'].strip()
    except KeyError:
        return False
    if ob != "" and eb != "" and sr != "" and classify_report_quality(cls_model, cls_tokenizer, str(report_dict), args):
        return True
    else:
        return False


def call_gen_model(gen_model, gen_tokenizer, cls_model, cls_tokenizer, data_item, args):
    def check_and_eval(gen_answer):
        re_pattern = r'(?<=</BUG REPORT>).*?(\{(?:[^{}]|(?R))*\})'
        re_match = regex.search(re_pattern, gen_answer.replace('\n', ''), regex.DOTALL | regex.MULTILINE)
        if re_match:
            try:
                gen_answer = eval(re_match.group(1))
                if check_new_report(cls_model, cls_tokenizer, gen_answer, args):
                    return gen_answer
            except:
                pass
        return dict()

    answer, gen_answer_text, call_cnt = dict(), "", 0
    while call_cnt < 5:
        gen_answer_text = call_model_to_gen(gen_model, gen_tokenizer, data_item['prompt'], args)
        call_cnt += 1
        answer = check_and_eval(gen_answer_text)
        if len(answer) != 0:
            break
    if call_cnt == 5 and len(answer) == 0:
        answer = gen_answer_text
        call_cnt = -1

    result = {
        'bug_id': data_item['bug_id'],
        'features': data_item['features'],
        'cnt': call_cnt,
        'answer': answer
    }
    return result


def run(args):
    """running """
    # 挂载logger
    args.logger = get_logger()
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 生成模型列表
    args.model_list = ['gpt-j', ]  # model list

    # 加载prompt数据
    prompt_data = load_prompt_data(args)

    # 加载分类模型
    cls_model, cls_tokenizer = load_classifier_model(args)

    # 遍历生成模型列表
    for model_name in args.model_list:

        # 加载生成模型
        args.gen_model_path = os.path.join(args.gen_model_dir, model_name)
        gen_model, gen_tokenizer = load_gen_model(args)

        # 执行五次生成实验
        for round_idx in range(0, 6):
            round_result_dict = dict()
            # 遍历每一个项目
            for project, datalist in prompt_data.items():  # traversal each project
                args.logger.info(f"===gen, model: {model_name}, pro: {project}, data num: {len(datalist)}===")
                # 遍历项目中的每一个report
                generated_list = []
                for data_item in tqdm(datalist):
                    args.logger.info(f"{model_name}, round: {round_idx}, {project}, {data_item['bug_id']}")
                    # 调用生成模型生成report
                    response = call_gen_model(gen_model, gen_tokenizer, cls_model, cls_tokenizer, data_item, args)
                    generated_list.append(response)
                round_result_dict[project] = generated_list
            args.logger.info(f"=======generated data, round: {round_idx}======")

            # 保存每一轮生成的数据
            with open(os.path.join(args.save_path, f'{model_name}_result_round_{round_idx}_new.json'), 'w') as f:
                json.dump(round_result_dict, f)

        args.logger.info(f"=======saved generated data, model: {model_name}======")


if __name__ == '__main__':
    parser = parse_run_arguments()
    run(parser)
