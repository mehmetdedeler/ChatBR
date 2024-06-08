# 利用LLM生成关键条目信息增强缺陷报告质量
***
# JavaUtils
> 处理原始缺陷报告文件工具类

# utils
> question1 baseline基线需要的工具类 Stanford NLP工具包

# readme
> README.md中的图片文件夹

# questions
> 本研究的所有RQ代码
## 1.question1
![img.png](readme/img1.png)
> + 使用预训练模型BERT构建一个更为有效的，准确率更高的缺陷报告分类模型。
> + bee-tool.py 基线工具
> + classifier.py 微调BERT模型的代码
> + data_augment.py 数据增强代码
> + xml2pkl.py 数据格式转换代码

## 2. question2
![img.png](readme/img2.png)
> + 筛选高质量缺陷报告（含有OB, EB, SR），并删除其中的元素，然后利用GPT生成缺失的信息，计算生成的信息和原始信息的语义相似度
> + pkl2json.py 数据格式转换代码
> + classifier_predict.py 预测缺陷报告质量，判断是否含有OB, EB, SR
> + analysis_sample.py 筛选符合GPT输入的缺陷报告，删除缺陷报告中的部分元素，构建数据集。
> + gpt_utils.py 调用GPT接口代码
> + run.py 整合数据处理 > GPT调用 > 数据分析流程的代码
> + eval_sim.py 使用Word2Vec计算GPT生成的信息和原始信息的语义相似度。(包括生成一个元素，多个元素。每个项目，所有项目的平均语义相似度)
> + plot.py 绘制数据分析结果图

## 3. question3
![img.png](readme/img3.png)
> + 计算生成有效信息的效率（生成一个正确的缺陷报告（OB, EB, SR,且格式正确）需要调用的GPT次数）
> + calculate_efficiency.py 计算每个特征，每个项目，所有特征，所有项目平均调用次数
> + plot.py 绘制数据分析结果图

## 4.question4
![img.png](readme/img4.png)
> + 尝试不同的Prompt对实验结果的影响
> + prompt.json 制定了三个不同的prompt
> + build_data.py 构建三个prompt的数据集
> + run.py 分别对三个不同prompt的数据集调用GPT，收集结果保存到generate_data文件夹
> + RQ4, prompt2 Birt没做完 2023.11.13

## 5. question5
![img.png](readme/img5.png)
> + 尝试不同LLM对实验结果的影响。使用question4中的第一个prompt（提示较为详细）在不同LLM上进行实验
> + run.py 分别使用GPT-3.5和GPT-4生成缺陷报告中的关键信息条目。生成的结果保存在generate_data文件夹。

## 6.question6(暂时先不做)
![img.png](readme/img6.png)
> + 尝试在下游任务（缺陷定位）验证GPT生成缺失信息的有效性。 从6个开源项目中获取3100+个缺陷报告，然后调用GPT-3.5生成缺失的关键条目信息。拿生成后的缺陷报告做基于信息检索的缺陷报告。
> + build_data.py 构建调用GPT-3.5需要的数据集
> + run.py 调用GPT-3.5生成新的缺陷报告
> + 缺陷定位的代码暂时还没写，需要找一个适合重构后的缺陷报告的缺陷定位的方法。
