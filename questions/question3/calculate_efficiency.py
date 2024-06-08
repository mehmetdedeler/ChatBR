"""计算平均调用LLM次数"""
import os


def calculate_llm_count():
    """计算所有缺失信息的情况，计算缺失一个元素，缺失两个元素,平均每生成一类信息需要调用的LLM次数"""
    for project in project_list:
        gen_filelist = os.listdir(f'../question2/generate_data/{project}')
        raw_filelist = os.listdir(f'../question2/predict_data/bert_new/{project}/selected_data')

        # 总的平均调用次数
        print(f"project: {project}, llm count: {len(gen_filelist) / (len(raw_filelist) * 6):.4f}")

        # 缺失部分信息的调用次数
        for combine_item in combine_list:
            gen_files = [file for file in gen_filelist if file.endswith(f"_remove_{combine_item}_gpt_")]
            print(f"project: {project}, feature: {combine_item}, avg_count: {len(gen_files) / len(raw_filelist):.4f}")


def analysis_invoke_count():
    """统计每一种缺失情况，调用LLM次数的占比"""
    for project in project_list:
        gen_filelist = os.listdir(f'../question7/round5/generate_data/prompt_2/{project}')
        # 特征组合列表
        for combine_item in combine_list:
            invoke_list = []
            # 调用五次
            for i in range(5):
                invoke_list.append(
                    len([file for file in gen_filelist if file.endswith(f"_remove_{combine_item}_gpt_{i}.json")])
                )
            # 计算次数
            for i in range(4):
                invoke_list[i] = invoke_list[i] - invoke_list[i+1]
            print(f"{invoke_list}")


if __name__ == '__main__':
    project_list = ["AspectJ", "Birt", "Eclipse", "JDT", "SWT", "Tomcat"]
    combine_list = ['OB', 'EB', 'SR', 'OB_EB', 'OB_SR', 'EB_SR']
    # calculate_llm_count()
    analysis_invoke_count()
