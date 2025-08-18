"""use bee-tool predict locate dataset whether contain OB, EB, SR"""

import json
import os
import re
import subprocess  # 导入subprocess模块，用于执行外部命令

import pandas as pd
import tqdm
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP

# 定义一个字典，用于存储单词和值
map = {}


def readWords():
    try:
        # 读取文件的内容
        import os
        # Get the directory where bee_tool.py is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dict_path = os.path.join(current_dir, 'model/bee/dict.txt')
        with open(dict_path, 'r', encoding='utf-8') as f:
            # 打印所有的行
            for line in f:
                line = line.rstrip()
                if line:  # 如果行不为空
                    index = line.rfind(':')  # 找到最后一个冒号的位置
                    key = line[:index]  # 截取冒号前面的部分作为键
                    value = line[index + 1:]  # 截取冒号后面的部分作为值
                    map[key] = value  # 将键值对存入字典

        print('Words were read: ' + str(len(map)))
    except Exception as err:
        print(err)


def isEmptySentence(sent_text):
    sent_text = re.sub(r'\s+|\u200B|\u200D|\uFEFF', "", sent_text)
    return sent_text == ""


def start_nlp():
    """启动stanford nlp工具包"""
    try:
        # Try to connect to existing server first using HTTP
        nlp = StanfordCoreNLP('http://localhost', port=9000)
        return nlp
    except Exception as e:
        print(f"Warning: Could not connect to existing Stanford CoreNLP server: {e}")
        print("Starting new Stanford CoreNLP server...")
        # Start a new server with different port to avoid conflicts
        nlp = StanfordCoreNLP('http://localhost', port=9001)
        return nlp


def close_nlp(nlp):
    """关闭stanford nlp工具包"""
    nlp.close()


def generateInputVector(sent, nlp):
    """定义一个函数，用于生成输入向量"""
    # 如果句子为空，抛出异常
    if not sent:
        raise ValueError("The sentence is empty")

    # Get annotation with error handling
    try:
        ann = nlp.annotate(sent, properties={
            'annotators': 'tokenize,ssplit,lemma,pos',  # Removed 'ner' to avoid TimeExpressionExtractorImpl error
            'outputFormat': 'json',
        })
        
        # Check if annotation is empty or None
        if not ann or ann.strip() == "":
            print(f"Warning: Empty annotation for sentence: '{sent}'")
            # Return a default feature vector
            return f"0 {len(map)}:{0}"
        
        # Parse JSON with error handling
        try:
            ann_parsed = json.loads(ann)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error for sentence: '{sent}'")
            print(f"Raw annotation: {repr(ann)}")
            # Return a default feature vector
            return f"0 {len(map)}:{0}"
        
        # Check if we have sentences and tokens
        if not ann_parsed.get('sentences') or len(ann_parsed['sentences']) == 0:
            print(f"Warning: No sentences in annotation for: '{sent}'")
            return f"0 {len(map)}:{0}"
        
        if not ann_parsed['sentences'][0].get('tokens') or len(ann_parsed['sentences'][0]['tokens']) == 0:
            print(f"Warning: No tokens in annotation for: '{sent}'")
            return f"0 {len(map)}:{0}"
        
        # 获取句子的词元和词性标签
        lemmas = [token['lemma'] for token in ann_parsed['sentences'][0]['tokens']]
        posTags = [token['pos'] for token in ann_parsed['sentences'][0]['tokens']]
        
    except Exception as e:
        print(f"Warning: Error in Stanford CoreNLP annotation for sentence: '{sent}'")
        print(f"Error: {e}")
        # Return a default feature vector
        return f"0 {len(map)}:{0}"

    # 创建一个空列表，用于存储特征值
    feature_array = []
    # 遍历词元和词性标签，生成一元和一阶词性特征，并更新计数器
    for i in range(len(lemmas)):
        unigram, tag_1 = lemmas[i], posTags[i]
        if map.get(unigram):
            feature_array.append(int(map.get(unigram)))
        if map.get(tag_1):
            feature_array.append(int(map.get(tag_1)))
    # 遍历词元和词性标签，生成二元和二阶词性特征，并更新计数器
    for i in range(len(lemmas) - 1):
        bigram = lemmas[i] + " " + lemmas[i + 1]
        tag_2 = posTags[i] + " " + posTags[i + 1]
        if map.get(bigram):
            feature_array.append(int(map.get(bigram)))
        if map.get(tag_2):
            feature_array.append(int(map.get(tag_2)))
    # 遍历词元和词性标签，生成三元和三阶词性特征，并更新计数器
    for i in range(len(lemmas) - 2):
        trigram = lemmas[i] + " " + lemmas[i + 1] + " " + lemmas[i + 2]
        tag_3 = posTags[i] + " " + posTags[i + 1] + " " + posTags[i + 2]
        if map.get(trigram):
            feature_array.append(int(map.get(trigram)))
        if map.get(tag_3):
            feature_array.append(int(map.get(tag_3)))

    # 对特征值进行排序，并格式化为svmlight的形式
    temp = "0"
    arr = sorted(list(set(feature_array)))
    # format the features for svmlight
    if len(arr) == 0:
        temp += f" {len(map)}:{0}"
    else:
        for i in range(len(arr)):
            temp += f" {arr[i]}:{1}"
            if i == len(arr) - 1 and arr[i] < len(map):
                temp += f" {len(map)}:{0}"

    return temp


def parseSentences(text, requestCounter):
    # print('parseSentences ' + requestCounter)

    modifiedSentences, originalSentences = [], []

    if isEmptySentence(text):
        return {'modifiedSentences': modifiedSentences, 'originalSentences': originalSentences}

    sentences = text.split("\n")

    # process insertCode, store the index of insertCode symbol
    symbolArray = []
    for i, sentence in enumerate(sentences):
        if '```' in sentence:
            symbolArray.append(i)
    indexOfInsertCode, setOfInsertCode, setOfOriginalInsertCode = [], [], []
    j = 0
    while j < (len(symbolArray) / 2):
        insertCode, originalInsertCode = "", ""
        m = j * 2
        begin, end = symbolArray[m], symbolArray[m + 1]
        indexOfInsertCode.append(end - begin + 1)
        for i, sentence in enumerate(sentences):
            if int(begin) <= i <= int(end):
                insertCode += sentence.replace('\n', '')
                originalInsertCode += sentence
        j = j + 1
        insertCode = insertCode.replace('```', '')  # remove the first two occurrences of ```
        setOfInsertCode.append(insertCode)
        setOfOriginalInsertCode.append(originalInsertCode)

    num, k = 0, 0
    while k < len(sentences):
        if '```' in sentences[k]:
            sentTxt = setOfInsertCode[num // 2]  # use integer division to get the index
            if not isEmptySentence(sentTxt):
                modifiedSentences.append(sentTxt)
                originalSentences.append(setOfOriginalInsertCode[num // 2])
            k = k + indexOfInsertCode[num // 2]
            num = num + 2
        elif sentences[k] != "":
            for sent in sent_tokenize((sentences[k])):
                new_sent = re.sub(r'[*#\->_]+|\[\s+\]|&nbsp;|\s{2,}', '', sent)
                if not isEmptySentence(new_sent):
                    modifiedSentences.append(new_sent)
                    originalSentences.append(sent)
            k = k + 1
        else:
            k = k + 1

    return {'modifiedSentences': modifiedSentences, 'originalSentences': originalSentences}


def getDefaultResponse():
    return {
        "code": 200,
        "status": 'success',
        "bug_report": {}
    }


def encode(sentences, originalSentences, stand_nlp):
    print('encode vector')
    indexOfLongString, sentVectors = [], []
    for i, sentence in enumerate(tqdm.tqdm(sentences, desc='encode iter', total=len(sentences))):
        origSentence = originalSentences[i]
        try:
            if len(origSentence) < 10000:
                sentVector = generateInputVector(sentence, stand_nlp)
                sentVectors.append(sentVector)
            else:
                print("sentence too long")
                indexOfLongString.append(i)
        except Exception as ex:
            print("Error encoding sentence: \"" + str(sentence) + "\"")
            print("Original sentence: \"" + origSentence + "\"")
            print(ex)
            # Instead of raising the exception, continue with a default vector
            print("Continuing with default feature vector...")
            sentVectors.append(f"0 {len(map)}:{0}")
    return {'sentVectors': sentVectors, 'indexOfLongString': indexOfLongString}


def getInputFileName(requestCounter):
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Sanitize the requestCounter to avoid spaces and special characters in filename
    safe_counter = re.sub(r'[^a-zA-Z0-9_-]', '_', str(requestCounter))
    return os.path.join(current_dir, f"input_{safe_counter}.dat")


def getOutputFile(prefix, requestCounter):
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Sanitize the requestCounter to avoid spaces and special characters in filename
    safe_counter = re.sub(r'[^a-zA-Z0-9_-]', '_', str(requestCounter))
    return os.path.join(current_dir, f"prediction_{prefix}_{safe_counter}.dat")


def writeVectors(sentVectors, requestCounter):
    print('writeVectors ' + requestCounter)
    inputFileName = getInputFileName(requestCounter)  # 调用getInputFileName函数，获取输入文件名

    sentVectors = "\n".join(sentVectors)
    with open(inputFileName, 'w') as f:  # 用追加模式打开文件
        f.write(sentVectors + "\n")  # 写入句子向量和换行符

    return inputFileName


def getResponse(sentences, obPrediction, ebPrediction, s2rPrediction, requestCounter, indexOfLongString):
    print('===getResponse ' + requestCounter)
    response = getDefaultResponse()  # 调用getDefaultRespose函数，获取默认的响应

    for k, sentence in enumerate(sentences):  # 遍历句子列表
        sentence = sentence.replace("\n", "")  # 去掉换行符

        labels = []  # 定义一个空列表，存储标签
        if k not in indexOfLongString:  # 如果句子的索引不在indexOfLongString中
            if float(obPrediction[k]) > 0:  # 如果obPrediction的值大于0
                labels.append("OB")  # 添加OB标签
            if float(ebPrediction[k]) > 0:  # 如果ebPrediction的值大于0
                labels.append("EB")  # 添加EB标签
            if float(s2rPrediction[k]) > 0:  # 如果s2rPrediction的值大于0
                labels.append("SR")  # 添加SR标签

        response["bug_report"][k] = {
            'text': sentence,
            'labels': labels
        }
    return response


def predict(prefix, inputFileName, requestCounter):
    print('predict' + prefix.lower() + " " + requestCounter)
    outputFile = getOutputFile(prefix.lower(), requestCounter)  # 调用getOutputFile函数，获取输出文件名
    # 构造执行命令
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Use the appropriate executable for the platform
    import platform
    if platform.system() == "Darwin":  # macOS
        svm_exe = os.path.join(current_dir, "model/bee/svm_classify")
    else:  # Windows
        svm_exe = os.path.join(current_dir, "model/bee/svm_classify.exe")
    model_file = os.path.join(current_dir, f"model/bee/model_{prefix}.txt")
    # Use list format to properly handle file paths with spaces
    command_list = [svm_exe, "-v", "1", inputFileName, model_file, outputFile]
    try:
        # if not os.path.exists(outputFile):
        #     try:
        #         print("touch new file: {}".format(outputFile))
        #         subprocess.run("touch {}".format(outputFile), check=True, shell=True)
        #     except subprocess.CalledProcessError as e:
        #         print(f'Error touch prediction file: {e}')
        subprocess.run(command_list, check=True)  # Use list format to handle spaces properly
    except subprocess.CalledProcessError as e:
        print(f'Error running SVM prediction: {e}')
        raise e

    with open(outputFile, 'r') as f:
        prediction = f.read().splitlines()

    return prediction


def removeFiles(requestCounter):
    print("removeFiles " + requestCounter)

    filesToRemove = [getInputFileName(requestCounter),
                     getOutputFile("ob", requestCounter),
                     getOutputFile("eb", requestCounter),
                     getOutputFile("sr", requestCounter)
                     ]

    for file in filesToRemove:  # 遍历需要删除的文件列表
        if os.path.exists(file):  # 如果文件存在
            os.remove(file)  # 删除文件
            print(file + ' was deleted')


def processText(bug_id, bug_text, stanford_nlp):
    requestCounter = bug_id
    try:
        # parse the sentences
        sentences = parseSentences(bug_text, requestCounter)
        if len(sentences['modifiedSentences']) == 0:
            return getDefaultResponse()

        # encode the sentences
        encodeResult = encode(sentences["modifiedSentences"], sentences["originalSentences"], stanford_nlp)
        sentVectors = encodeResult["sentVectors"]
        indexOfLongString = encodeResult["indexOfLongString"]

        # write the sentences to a file
        inputFileName = writeVectors(sentVectors, requestCounter)

        # predict the output
        obPrediction = predict("OB", inputFileName, requestCounter)
        ebPrediction = predict("EB", inputFileName, requestCounter)
        srPrediction = predict("SR", inputFileName, requestCounter)

        # read the prediction
        return getResponse(sentences["originalSentences"], obPrediction, ebPrediction, srPrediction, requestCounter,
                           indexOfLongString)
    except Exception as err:
        print('There was an error: ' + str(err))
        raise err
    finally:
        removeFiles(requestCounter)


def test_aug_data(stanford_nlp):
    # 计算bee-tool认为增强后的数据有多少样本的标签发生了改变
    requestCounter = "1"
    try:
        # parse the sentences
        df = pd.read_pickle('./all_statement_aug.pkl')

        print("===before data length : {}===".format(len(df['text'].tolist())))

        df['text'] = df['text'].apply(lambda x: re.sub(r'[*#\->_]+|\[\s+\]|&nbsp;|\s{2,}', '', x))
        df = df[df['text'] != '']
        df = df.reset_index(drop=True)

        sentences = {
            "modifiedSentences": df['text'].tolist(),
            "originalSentences": df['text'].tolist()
        }

        print("===after data length : {}===".format(len(df['text'].tolist())))

        if len(sentences['modifiedSentences']) == 0:
            return getDefaultResponse()

        # encode the sentences
        encodeResult = encode(sentences["modifiedSentences"], sentences["originalSentences"], stanford_nlp)
        sentVectors = encodeResult["sentVectors"]
        indexOfLongString = encodeResult["indexOfLongString"]

        # write the sentences to a file
        inputFileName = writeVectors(sentVectors, requestCounter)

        # predict the output
        obPrediction = predict("OB", inputFileName, requestCounter)
        ebPrediction = predict("EB", inputFileName, requestCounter)
        srPrediction = predict("SR", inputFileName, requestCounter)

        # read the prediction
        result = getResponse(sentences["originalSentences"], obPrediction, ebPrediction, srPrediction, requestCounter,
                             indexOfLongString)

        bert_labels_list = df['labels'].tolist()
        cnt = 0
        if len(result['bug_report'].values()) == len(bert_labels_list):
            bee_rst_list = []
            for idx, item in enumerate(result['bug_report'].values()):
                label_item = [0, 0, 0]
                if 'OB' in item['labels']:
                    label_item[0] = 1
                elif 'EB' in item['labels']:
                    label_item[1] = 1
                elif 'SR' in item['labels']:
                    label_item[2] = 1
                bee_rst_list.append([item['text'], label_item])
                if df.loc[idx:idx, 'text'].iloc[0] != item['text']:
                    print("==bee和bert行文本不一致==")
                else:
                    df.loc[idx:idx, 'bee_label'] = str(label_item)
                    if str(label_item) != str(bert_labels_list[idx]):
                        cnt += 1

            print("bert和bee预测样本标签不一样比例：{:.4f}".format(cnt / len(bert_labels_list)))
            bee_df = pd.DataFrame(bee_rst_list, columns=['text', 'label'])
            bee_df.to_csv('./bee_rst.csv')
            df.to_csv('./bee_bert_result.csv')
        else:
            raise RuntimeWarning("数据长度不一致！")

    except Exception as err:
        print('There was an error: ' + str(err))
        raise err
    finally:
        removeFiles(requestCounter)


def test_one_sample():
    """测试单条数据"""
    bug_report = {
        "bug_id": "101039",
        "title": "Bug 101039 Series colors do not have different default values",
        "description": "When you create a pie chart, all colors are gray by default."
                       "When you create a bar chart with multiple Y series, additional Y series are all gray by default."
                       "Expected: The pie chart should have different colors for each value by default. "
                       "For bar charts, the additional series should be automatically be assigned a different color by default."
    }

    readWords()
    core_nlp = start_nlp()
    result = processText(bug_report['bug_id'], bug_report['title']+'\n'+bug_report['description'], core_nlp)
    core_nlp.close()

    print("===bee tool predict result===")
    print(result)
    # save bug report
    save_file_path = '../question2/predict_data/bee-tool'
    if not os.path.exists(save_file_path):
        os.mkdir(save_file_path)
    with open(os.path.join(save_file_path, bug_report['bug_id']+'.json'), 'w') as f:
        json.dump(result, f)


def test_multiple_data():
    """测试多条数据"""
    readWords()
    core_nlp = start_nlp()

    # dir_list = os.listdir('./dataset')
    dir_list = ["AspectJ", "Eclipse", "JDT", "SWT", "Tomcat"]
    for dir_item in dir_list:
        perfect_cnt = 0  # 统计项目中有多少同时含有ob, eb, sr的样本
        dir_path = os.path.join('../question2/origin_data', dir_item)
        filelist = os.listdir(dir_path)

        p_bar = tqdm.tqdm(filelist, desc="iter:", total=len(filelist))
        for idx, filename in enumerate(p_bar):

            p_bar.set_description(f"{dir_item}: No.{idx}")

            if filename.split(".")[-1] == 'json':
                with open(os.path.join(dir_path, filename), 'r') as f:
                    report = json.load(f)
                f.close()

                # parse bug report
                try:
                    result = processText(report['bug_id'], report['title'] + '.\n' + report['description'], core_nlp)
                except:
                    continue

                # save bug report
                save_path = os.path.join('../question2/predict_data/bee-tool', dir_item)
                perfect_save_path = os.path.join(save_path, 'perfect_data')  # 同时含有ob, eb, sr样本存放路径
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if not os.path.exists(perfect_save_path):
                    os.mkdir(perfect_save_path)

                # 保存bee-tool预测结果文件
                with open(os.path.join(save_path, filename), 'w') as f:
                    json.dump(result, f)
                f.close()

                # 如果report含有ob, eb, sr
                label_list = []
                for sent in result['bug_report'].values():
                    label_list.extend(sent['labels'])

                if len(set(label_list)) == 3:  # OB, EB, SR
                    perfect_cnt += 1
                    with open(os.path.join(perfect_save_path, filename), 'w') as f:
                        json.dump(result, f)
                    f.close()
        print("==={} has {} perfect sample===".format(dir_item, perfect_cnt))

    core_nlp.close()


def calculate_aug_data():
    """计算bee-tool认为增强后的数据有多少样本的标签发生了改变"""
    readWords()
    core_nlp = start_nlp()
    test_aug_data(core_nlp)
    core_nlp.close()


if __name__ == '__main__':

    test_one_sample()

    # text = """Bug 101187Localization key can't be blank when adding. Description:If key is blank, externalized text can't be displayed in chart.Steps to reproduce:1."""
    #
    # core_nlp = start_nlp()
    # doc = core_nlp.annotate(text, properties={
    #     'annotators': 'tokenize',
    #     'outputFormat': 'json',
    # })
    # print(doc)
    # core_nlp.close()

