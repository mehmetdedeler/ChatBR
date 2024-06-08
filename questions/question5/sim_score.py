import json
import os
import gensim
import pandas as pd
from nltk import word_tokenize
from scipy import spatial

from questions.log_utils import get_logger


class SimilarityScore(object):

    def __init__(self, embedding_path, base_path, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.result_data_path = None
        self.gen_data = {}
        self.base_data = {}
        self.logger = get_logger()
        self.projects = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
        self.model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        self.load_base_data(base_path)
        self.model_list = ['llama2', 'vicuna-7b', 'incoder-6b']
        self.project_result = {}

    def load_generated_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            self.gen_data = json.load(file)

    def load_base_data(self, file_path):
        for project in self.projects:
            self.base_data[project] = {}
            filelist = os.listdir(os.path.join(file_path, project, 'selected_data'))
            for filename in filelist:
                if filename.endswith('.json'):
                    with open(os.path.join(file_path, project, 'selected_data', filename), 'r') as file:
                        report_data = json.load(file)
                    self.base_data[project][report_data['id']] = report_data

    def calculate_sim_use_word2vec(self):
        for model_name in self.model_list:
            self.project_result[model_name] = {round_idx: {} for round_idx in range(1, 6)}
            for round_idx in range(1, 6):
                self.load_generated_data(os.path.join(self.data_path, f'{model_name}_result_round_{round_idx}_new.json'))
                for project, datalist in self.gen_data.items():
                    self.project_result[model_name][round_idx][project] = self.parse_data_list(project, datalist)
        self.calculate_average()
        self.save_result()
        # 计算平均值
        self.calculate_mean_score()

    def parse_data_list(self, project, datalist):
        project_result_list = []
        for gen_data in datalist:
            base_report = self.base_data[project][gen_data['bug_id']]
            # re_match = re.findall(r'>(\{[^}]*\})<', gen_data['answer'].replace('\n', ''), re.DOTALL)
            # try:
            #     matched_report = eval(re_match[-1]) if re_match and len(re_match) > 1 else {}
            # except SyntaxError:
            #     matched_report = {}
            for feature in gen_data['features']:
                score = self.calculate_score(base_report, gen_data['answer'], feature)
                project_result_list.append([gen_data['bug_id'], '_'.join(gen_data['features']), feature, score])
        return project_result_list

    def calculate_score(self, base_report, report_answer, feature):
        try:
            return 1 - spatial.distance.cosine(
                self.model.get_mean_vector(word_tokenize(base_report[feature])),
                self.model.get_mean_vector(word_tokenize(report_answer.get(feature)))
            )
        except (ValueError, AttributeError, TypeError):
            return 0

    def calculate_average(self):
        temp_data = {}
        for model_name, model_data in self.project_result.items():
            temp_data[model_name] = {}
            for project, project_data in model_data[1].items():
                averages = [sum(data[-1] for data in project_data) / 5 for project_data in zip(*[model_data[idx][project] for idx in range(1, 6)])]
                temp_data[model_name][project] = [[*data[:-1], avg] for data, avg in zip(project_data, averages)]
        self.project_result = temp_data.copy()

    def save_result(self):
        with open(os.path.join(self.save_path, 'result.json'), 'w') as f:
            json.dump(self.project_result, f)

    def calculate_mean_score(self):
        with open(os.path.join(self.save_path, 'result.json'), 'r') as f:
            data = json.load(f)
        result_json = dict()
        for model, model_data in data.items():
            model_result = dict()
            for project, project_data in model_data.items():
                df = pd.DataFrame(project_data, columns=['bug_id', 'features', 'feature', 'score'])
                grouped_df = df.groupby(['features', 'feature'])['score'].mean().reset_index().values.tolist()
                for item in grouped_df:
                    self.logger.info(
                        f"model: {model}, project: {project}, features: {item[0]}, feature: {item[1]}, score: {item[2]}"
                    )
                model_result[project] = grouped_df
            result_json[model] = model_result
        with open(os.path.join(self.save_path, 'score_result.json'), 'w') as f:
            json.dump(result_json, f)


# def format_result_json():
#     """load result json"""
#     new_format_result = {}
#     with open('./generate_data/llama2_result_round_1_back.json', 'r') as f:
#         data = json.load(f)
#     for project, datalist in data.items():
#         new_format_result[project] = []
#         for data_item in datalist:
#             for filename, answer in data_item.items():
#                 re_match = re.findall(r'>(\{[^}]*\})<', answer.replace('\n', ''), re.DOTALL)
#                 try:
#                     answer = eval(re_match[-1]) if re_match and len(re_match) > 1 else {}
#                 except SyntaxError:
#                     answer = {}
#                 new_format_result[project].append({
#                     'bug_id': filename.split('_')[0],
#                     'features': filename.split('remove_')[1].split('_'),
#                     'answer': answer
#                 })
#     with open('./generate_data/llama2_result_round_1.json', 'w') as f:
#         json.dump(new_format_result, f)
#
#
# def format_result_to_json():
#     """load result json"""
#     for i in range(1, 6):
#         with open(f'./generate_data/incoder-6b_result_round_{i}.json', 'r') as f:
#             data = json.load(f)
#         new_format_result = {}
#         for project, datalist in data.items():
#             new_format_result[project] = []
#             for answer in datalist:
#                 match = re.search(r'<BUG REPORT>(.*?)</BUG REPORT>', answer, re.DOTALL)
#                 re_match = re.findall(r'>(\{[^}]*\})<', answer.replace('\n', ''), re.DOTALL)
#                 try:
#                     answer = eval(re_match[-1]) if re_match and len(re_match) > 1 else {}
#                 except SyntaxError:
#                     answer = {}
#                 if match:
#                     report_json_format = eval(match.group(1).strip())
#                     empty_keys = [key for key in ('OB', 'EB', 'SR') if report_json_format.get(key) == '']
#                     if len(empty_keys) > 0:
#                         new_format_result[project].append({
#                             'bug_id': report_json_format['id'],
#                             'features': empty_keys,
#                             'answer': answer
#                         })
#             with open(f'./generate_data/back/vicuna-7b_result_round_{i}.json', 'w') as f:
#                 json.dump(new_format_result, f)


if __name__ == '__main__':
    device = "local"
    if device == 'local':
        similarity = SimilarityScore(embedding_path='../../utils/embed_model/GoogleNews-vectors-negative300.bin',
                                     base_path='../question2/predict_data/bert_new',
                                     data_path='./generate_data/',
                                     save_path='./result_data/')
    else:
        similarity = SimilarityScore(embedding_path='../../utils/embed_model/GoogleNews-vectors-negative300.bin',
                                     base_path='/Se-jiwangjie/re_report/questions/question2/predict_data/bert_new',
                                     data_path='/Se-jiwangjie/re_report/questions/question5/generate_data/',
                                     save_path='/Se-jiwangjie/re_report/questions/question5/result_data/')
    similarity.calculate_sim_use_word2vec()
