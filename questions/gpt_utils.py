import json
import re

import openai

openai.api_key = "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"


def call_ChatGPT(bug_report, model="gpt-3.5-turbo"):
    """调用ChatGPT"""
    prompt_match = re.search(r'(.*?)<BUG REPORT>', bug_report, re.DOTALL)
    report_match = re.search(r'<BUG REPORT>.*?</BUG REPORT>', bug_report, re.DOTALL)
    if prompt_match and report_match:
        llm_prompt = prompt_match.group(1).strip()
        input_report = report_match.group(0)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": llm_prompt},
                    {"role": "user", "content": input_report},
                ]
            )
        except:
            print("=====call ChatGPT error=====")
            return 1, None

        # 解析数据
        if response["choices"][0]["finish_reason"] == "length":
            print("====The output exceeds the length limit====")
            return 2, None
        elif response["choices"][0]["finish_reason"] == "stop":
            # response["origin"] = {}
            # response["origin"]["input_prompt"] = llm_prompt
            # response["origin"]["input_content"] = input_report.split('</BUG REPORT>')[0].strip()
            response = response["choices"][0]["message"]["content"]
            response = response.encode('utf-8', 'ignore').decode("utf-8")  # 去除非utf8字符
            return 0, response
        else:
            print("====invoke openai api error====")
            return 3, None
    else:
        print("===parse response error===")
