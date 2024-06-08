import json
import os

from classifier_predict import is_report_perfect, bert_parse_arguments, create_bert_model
from questions.log_utils import get_logger


def evaluate_gpt_generate(data_path):
    """分析生成的缺陷报告是否含有OB, EB, SR"""
    args = bert_parse_arguments()
    args.model, args.tokenizer = create_bert_model(args)

    project_list = ["Birt"]
    for project in project_list:
        cnt = 0
        gen_filelist = os.listdir(os.path.join(data_path, project))
        base_project_path = f"../question7/prompt_data/prompt_2/{project}"
        prompt_filelist = [filename.strip(".txt") for filename in os.listdir(base_project_path)]

        for prompt_file in prompt_filelist:
            try:
                filename = [file for file in gen_filelist if file.startswith(prompt_file)][-1]
                with open(os.path.join(data_path, project, filename), 'r') as f:
                    report = json.load(f)
                report_text = " ".join(report.values())
            except Exception as e:
                logger.info(f"error==> {project} ==> filename: {prompt_file} ==> report_labels: {e}")
                cnt += 1
                continue

            labels_set = is_report_perfect(report_text, args)
            if len(labels_set) != 3:
                cnt += 1
                logger.info(f"{project} ==> filename: {filename} ==> report_labels: {','.join(labels_set)}")

        logger.info(f"project: {project} has {cnt} not perfect report")


if __name__ == '__main__':
    logger = get_logger()
    evaluate_gpt_generate("../question7/round1/generate_data/prompt_2/")
