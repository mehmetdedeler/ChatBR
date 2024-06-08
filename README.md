# ChatBR: Automated assessment and improvement of bug report quality using ChatGPT
***
# abstract
Bug reports, containing crucial information such as the Observed Behavior (OB), Expected Behavior (EB), and Steps to Reproduce (S2R), can help developers reproduce and fix bugs efficiently. However, due to the limited experience of the some report writer and the complexity of some bugs, the bug reports often miss this crucial information.  Although some machine learning based approaches and information retrieval (IR) based techniques are proposed to detect and supplement the missing information in bug reports, the performance of these approaches depends heavily on the size and quality of the bug report dataset.

The development of fine-tuning pre-trained models and LLMs can effectively alleviate the problems.  In this paper, we present **ChatBR**, a method for automated assessment and improvement of bug report quality using ChatGPT.First, we fine-tune a pre-trained BERT model using manually annotated bug reports to create a statement-level multi-label classifier to assess the quality of bug reports by detecting whether existing OB, EB, and S2R. Then, we use ChatGPT in a zero-shot setup to generate missing information (OB, EB, and S2R) to improve the quality of the bug report. Finally, we manually check the consistency of the output of ChatGPT with that of the classifier with high confidence. Experimental results show that, in the task of detecting missing information in bug reports,  **ChatBR**  outperforms the state-of-the-art methods by 25.38\%-29.20\%  in terms of precision. In the task of generating missing information in bug reports, **ChatBR** can achieve an average of 84.10\% in terms of semantic similarity of the generated information across six different projects. Furthermore, **ChatBR** can generate more than 99.9\% of high quality bug reports (i.e., bug reports that are full of OB, EB, and S2R) within five calls to ChatGPT.

# OverView
![image](https://github.com/jiwangjie/Ultra/assets/74883729/95e7b753-64e2-460e-b122-55de118b89c4)

# JavaUtils
> Tools for managing unprocessed defect report files

# utils
> question1 baseline Required tools Stanford NLP tools

# questions
> All RQ codes for this study
## 1.question1
> + A more efficient and accurate defect report classification model is constructed by using pre-trained model BERT.
> + bee-tool.py Baseline tool
> + classifier.py Fine-tune the BERT model code
> + data_augment.py Data enhancement code
> + xml2pkl.py Data format conversion code

## 2. question2
> + Filter high-quality defect reports (OB, EB, SR) and remove the elements, then use GPT to generate the missing information and calculate the semantic similarity between the generated information and the original information
> + pkl2json.py Data format conversion code
> + classifier_predict.py Predict defect report quality and determine whether OB, EB, SR is present
> + analysis_sample.py Filter the defect reports that match the GPT input, remove some elements from the defect report, and build the dataset.
> + gpt_utils.py Call the GPT interface code
> + run.py Code that integrates data processing > GPT calls > Data analysis processes
> + eval_sim.py The semantic similarity between the GPT generated information and the original information is calculated using Word2Vec. (This includes generating one element, multiple elements. Per item, average semantic similarity across all items)
> + plot.py Plot data analysis results (discarded)

## 3. question3
> + Calculate the efficiency of generating valid information (the number of GPT calls required to generate a correct defect report (OB, EB, SR, and properly formatted))
> + calculate_efficiency.py Calculate the average number of calls per feature, per item, all features, all items
> + plot.py Plot the results of data analysis

## 4.question4 & question7
> + Try the effect of different Prompt on experimental results
> + prompt.json Three different prompts were formulated
> + build_data.py Build three prompt datasets
> + run.py Call GPT on three different prompt datasets and save the collected results to the generate_data folder

## 5. question5 & question6
> + Try the influence of different LLM on experimental results. Experiment on different LLMS using the first prompt in question4 (which is more detailed)
> + run.py uses GPT-3.5 and llama2-7b, respectively, and vicuna-7b to generate key information entries in defect reports. The generated results are saved in the generate_data folder.

