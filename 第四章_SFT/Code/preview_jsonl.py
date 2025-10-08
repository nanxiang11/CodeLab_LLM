import json

file_path = "/root/autodl-tmp/Dataset/BelleGroup/train_3.5M_CN.json"
k = 1000

def stream_json_objects(file_path, max_items=1000):
    with open(file_path, "r", encoding="utf-8") as f:
        buf = []
        depth = 0
        in_string = False
        escape = False
        count = 0
        while True:
            ch = f.read(1)
            if not ch:
                break
            buf.append(ch)
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        raw = ''.join(buf)
                        buf = []
                        try:
                            obj = json.loads(raw)
                            yield obj
                            count += 1
                            if count >= max_items:
                                return
                        except json.JSONDecodeError:
                            print("解析失败：", raw[:100])

# 使用
for item in stream_json_objects(file_path, k):
    print(item)
