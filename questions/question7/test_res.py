import json


def run():
    with open("./round1/generate_data/prompt_0/Eclipse/7581_remove_EB_gpt_1.json", 'r') as f:
        res_json = json.load(f)
        try:
            response_text = ' '.join(res_json)
            # 检查OB, EB ,SR不为空且含有OB, EB, SR标签
            if len(res_json['OB'].strip()) > 5 and len(res_json['EB'].strip()) > 5 and len(res_json['SR'].strip()) > 5 \
                    :
                return True
            else:
                print('<====generated report is no perfect report====>')
                return False
        except:
            print(f"<====generated report format error====>")
            return False


if __name__ == '__main__':
    print(run())
