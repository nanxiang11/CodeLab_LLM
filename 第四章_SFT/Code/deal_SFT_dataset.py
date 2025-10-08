import json
from tqdm import tqdm

sft_data = '/root/autodl-tmp/Dataset/BelleGroup/train_3.5M_CN.json'
output_sft_data = '/root/autodl-tmp/Dataset/BelleGroup/BelleGroup_sft.jsonl'


def convert_message(data):
    """
    将原始数据转换为标准格式
    """
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

with open(output_sft_data, 'a', encoding='utf-8') as sft:
    with open(sft_data, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for item in tqdm(data, desc="Processing", unit="lines"):
            item = json.loads(item)
            message = convert_message(item['conversations'])
            sft.write(json.dumps(message, ensure_ascii=False) + '\n')