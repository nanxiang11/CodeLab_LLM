from transformers import AutoTokenizer




def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 测试基本属性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一名住在南巷的居民，熟悉小巷生活和街坊习惯。"},
        {"role": "user", "content": "我今天在南巷遇到一只花猫，好可爱！"},
        {"role": "assistant", "content": "太棒了！南巷的花猫很友好，你给它取名字了吗？"},
        {"role": "user", "content": "还没有，你有什么建议吗？"},
        {"role": "assistant", "content": "可以叫它‘巷巷’，听起来很有南巷的感觉。"},
    ]

    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # 测试编码解码
    print("\n=== 编码解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    print("Encoded input_ids:", encoded["input_ids"])
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # 测试特殊token处理
    print("\n=== 特殊token处理 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)


eval_tokenizer('/root/StudyLLM/NX_LLM/第二章 动手实现/tokenizer_k')