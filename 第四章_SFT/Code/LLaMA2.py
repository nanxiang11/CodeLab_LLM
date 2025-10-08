from typing import Optional

import math
import torch
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F

from ModelConfig import ModelConfig
from DecoderLayer import DecoderLayer
from RMSNorm import RMSNorm
from ROPE import *


class Transformer(PreTrainedModel, GenerationMixin):
    """
        自回归 Transformer 模型（类似 GPT/LLaMA 架构）
        - 支持多层 Decoder
        - 使用 Rotary Positional Embedding
        - 可共享词嵌入与输出投影
        - 输出 CausalLMOutputWithPast 类型
    """
    # 注入配置类
    config_class = ModelConfig
    # 记录最后一次计算的损失
    last_loss: Optional[torch.Tensor]

    def __init__(self, args: ModelConfig):
        super().__init__(config=args)

        # 初始化模型参数
        self.args = args
        # 词汇表大小,这个大小必须是和之前我们训练的Tokenizer里面一样，不然id会错乱
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers

        # 词嵌入层，这一步是将我们id变为向量的形式，这里可以联想一下独热编码
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout层
        self.dropout = nn.Dropout(args.dropout)

        # Decoder层，这里通过短短的三行代码就实现对Decoder层的遍历嵌套
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))

        # 归一化层
        self.rmsnorm = RMSNorm(args.dim, eps=args.norm_eps)

        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight

        # 相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_rotary_freqs(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # 输出容器
        self._no_split_modules = [name for name, _ in self.named_modules()]  # 不分割的模块列表

    def _init_weights(self, module):
        # 初始化权重的函数
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens=None, targets=None, input_ids=None, attention_mask=None, **kwargs):
        # 如果通过关键字传入了 input_ids，就用它
        if tokens is None and input_ids is not None:
            tokens = input_ids
        if targets is None and attention_mask is not None:
            targets = attention_mask

        if tokens is None:
            raise ValueError("No input tokens provided")

        # 下面保持原 forward 实现
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.rmsnorm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0,
                                             reduction='none')
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    # ================= LoRA / PEFT 兼容接口 =================
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        PEFT 生成调用接口
        """
        return {"tokens": input_ids, **kwargs}

    def get_input_embeddings(self):
        return self.tok_embeddings

    def get_output_embeddings(self):
        return self.output

    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        逐步生成序列（采样版本）
            - idx: [batch_size, seq_len] 输入序列
            - stop_id: 停止 token id
            - max_new_tokens: 最大生成长度
            - temperature: 采样温度
            - top_k: top-k 采样
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond).logits
            logits = logits[:, -1, :]  # 只保留最后一个时间步的输出

            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == stop_id:
                break

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        # 只返回生成的token
        return idx[:, index:]



if __name__ == "__main__":

    def count_parameters(model):
        """
        统计模型中可训练参数的数量

        Args:
            model: PyTorch模型

        Returns:
            int: 可训练参数总数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    args = ModelConfig()
    # LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
    x = torch.randint(0, 6144, (64, 256))  # [bs, seq_len]
    # 实例化LLaMA2Model
    model = Transformer(args=args)
    # 计算model的全部参数
    print(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')

    out = model(x)
    print(out.logits.shape)  # [batch_size, 1, vocab_size]