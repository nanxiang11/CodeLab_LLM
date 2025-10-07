import torch.nn.functional as F
import math
import torch
from torch import nn
from torch.nn.modules.module import T
from ROPE import *
from ModelConfig import ModelConfig

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 key/value 的头，保证与 query 的头数对齐

    x: [batch, seq_len, n_kv_heads, head_dim]
    n_rep: 重复次数
    """
    if n_rep == 1:
        return x  # 不需要重复，直接返回

    bs, slen, n_kv_heads, head_dim = x.shape

    # 在头维度后面加一个维度
    x = x[:, :, :, None, :]  # [bs, seq_len, n_kv_heads, 1, head_dim]

    # 扩展这个维度到 n_rep
    x = x.expand(bs, slen, n_kv_heads, n_rep, head_dim)  # [bs, seq_len, n_kv_heads, n_rep, head_dim]

    # 合并 head 维度和重复维度
    x = x.reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # [bs, seq_len, n_kv_heads*n_rep, head_dim]

    return x



# if __name__ == "__main__":
#
#     # 假设 batch=1, seq_len=2, n_kv_heads=2, head_dim=3
#     x = torch.tensor([
#         [
#             [[1,2,3], [4,5,6]],    # 序列位置 0
#             [[7,8,9], [10,11,12]]  # 序列位置 1
#         ]
#     ], dtype=torch.float)
#
#     print("原始 x:")
#     print(x)
#     print("shape:", x.shape)  # [1, 2, 2, 3]
#
#     # 重复每个 head 2 次
#     n_rep = 2
#     x_repeated = repeat_kv(x, n_rep)
#
#     print("\n重复后的 x:")
#     print(x_repeated)
#     print("shape:", x_repeated.shape)  # [1, 2, 4, 3]



class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        # 我们更具有多少个头的数量来确定需要多少个key 和 value的数量
        # 如果没有指定 key/value 的头数，就默认使用 query 的头数
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else self.n_heads

        # 确保总头数可以被键值头数整除。
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理大小，默认为1。model_parallel_size=1 表示 没有分割，单卡计算如果你用多卡训练，模型会被切分成多个部分并行计算
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值头数除以模型并行处理大小。
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键和值的尺寸。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # 开始定义qkv的权重矩阵W这个部分和传统的并未差别
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)

        # 最后合并输出权重矩阵，这个时候其实就是说明了wq，wo他们的维度是一样的。
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout，防止过拟合，这里分为两个，一个是注意力计算过程中的，一个是输出矩阵中的
        self.attn_dropout = nn.Dropout(args.dropout)
        self.output_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 上面我们都定义好了，开始到计算环节了，这里我们环境是满足Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask，这个手动实现也是非常简单的
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，用于遮蔽未来信息。这个-inf就已经代表负无穷。
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    # 开始真正的撸流程了，按照框架图一步一步的编写就行。
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        """
        前向传播计算注意力。
        x: [batch_size, seq_len, dim] 输入特征
        freqs_cos, freqs_sin: 旋转位置嵌入（RoPE）预计算的 cos 和 sin
        """

        # 获取输入形状
        batch_size, seq_len, dim = x.shape
        # xq/xk/xv 要求 shape: [batch, seq_len, n_heads, head_dim]

        # -----------------------------
        # 计算 Q/K/V 矩阵
        # -----------------------------
        # 通过线性变换生成查询（Q）、键（K）、值（V）
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 调整形状以适应多头注意力
        # Q 头数 = n_local_heads
        xq = xq.reshape(batch_size, seq_len, self.n_local_heads, self.head_dim)
        # K/V 头数 = n_local_kv_heads
        xk = xk.reshape(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(batch_size, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入（RoPE）
        # 将 Q/K 的向量旋转编码位置信息，使模型可以感知序列位置
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 扩展 K/V 头以匹配 Q 头
        # 当 KV 头数 < Q 头数时，需要重复 KV 头
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 转置为 [batch, heads, seq_len, head_dim]
        # 方便矩阵乘法和注意力计算
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 计算注意力
        if self.flash:
            # 如果支持 Flash Attention（PyTorch >= 2.0），使用高效实现
            # is_causal=True 保证未来信息不可见（自回归）
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # 手动实现注意力计算（慢版本）
            # 计算注意力分数: QK^T / sqrt(head_dim)
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            # 添加 causal mask 遮蔽未来信息
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seq_len, :seq_len]
            # softmax 得到注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # dropout 防止过拟合
            scores = self.attn_dropout(scores)
            # 加权求和得到输出
            output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头
        # output shape: [batch, heads, seq_len, head_dim] -> [batch, seq_len, heads*head_dim]
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)

        # 输出投影回模型维度并 dropout
        output = self.wo(output)
        output = self.output_dropout(output)

        return output





# if __name__ == "__main__":
#     args = ModelConfig()
#     # 创建Attention实例
#     attention_model = Attention(args)
#
#     # 模拟输入数据
#     batch_size = 1
#     seq_len = 50  # 假设实际使用的序列长度为50
#     dim = args.dim
#     x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
#
#     freqs_cos, freqs_sin = precompute_rotary_freqs(dim // args.n_heads, seq_len)
#
#     # 运行Attention模型
#     output = attention_model(x, freqs_cos, freqs_sin)
#
#     # attention出来之后的形状 依然是[batch_size, seq_len, dim]
#     print("Output shape:", output.shape)



if __name__ == "__main__":
    import os
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import math
    import torch.nn.functional as F

    # ---------- 绘图工具 ----------
    def _to_numpy(t: torch.Tensor) -> np.ndarray:
        """把 tensor 转成 numpy（先 detach、cpu）。"""
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.array(t)

    def plot_attention_matrix(weights, batch_idx=0, head_idx=0, title=None, show=True, fname=None):
        """
        画单个 head 的注意力矩阵（heatmap）。
        weights: [B, H, L, L] 的 numpy 或 torch tensor
        """
        w = _to_numpy(weights)
        mat = w[batch_idx, head_idx]  # [L, L]
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, aspect="auto")
        plt.xlabel("Key position")
        plt.ylabel("Query position")
        plt.title(title or f"Batch {batch_idx} Head {head_idx}")
        plt.colorbar()
        if fname:
            plt.savefig(fname, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def plot_avg_attention(weights, batch_idx=0, title=None, show=True, fname=None):
        """
        所有 head 平均后的 attention heatmap。
        """
        w = _to_numpy(weights)
        mat = w[batch_idx].mean(axis=0)  # [L, L]
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, aspect="auto")
        plt.xlabel("Key position")
        plt.ylabel("Query position")
        plt.title(title or f"Batch {batch_idx} AvgHeads")
        plt.colorbar()
        if fname:
            plt.savefig(fname, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def plot_attention_for_query_token(weights, query_pos, batch_idx=0, head_idx=0, show=True, fname=None):
        """
        绘制某个 query_pos 的 attention distribution（对所有 key positions）。
        weights: [B, H, L, L]
        """
        w = _to_numpy(weights)
        vec = w[batch_idx, head_idx, query_pos]  # 长度 L
        plt.figure(figsize=(8, 2))
        plt.bar(np.arange(len(vec)), vec)
        plt.xlabel("Key position")
        plt.ylabel("Attention weight")
        plt.title(f"Batch {batch_idx} Head {head_idx} QueryPos {query_pos}")
        if fname:
            plt.savefig(fname, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    # ---------- 计算 attention weights（slow path，用于可视化） ----------
    @torch.no_grad()
    def compute_attn_weights_for_vis(xq: torch.Tensor, xk: torch.Tensor, mask: torch.Tensor | None = None, head_dim: int | None = None) -> torch.Tensor:
        """
        输入：
            xq, xk: [B, H, L, D]
            mask: optional causal mask with shape broadcastable to [B, H, L, L] (can be None)
            head_dim: D，若为 None 则从 xq.shape[-1] 读取
        返回：
            weights: [B, H, L, L]（CPU tensor，便于绘图）
        """
        if head_dim is None:
            head_dim = xq.shape[-1]

        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(head_dim)  # [B,H,L,L]

        if mask is not None:
            # 确保 mask 与 scores 在同一设备 / dtype，然后相加
            if mask.device != scores.device:
                mask = mask.to(scores.device)
            scores = scores + mask

        weights = F.softmax(scores.float(), dim=-1)
        return weights.detach().cpu()

    # ---------- 准备模型与输入 ----------
    args = ModelConfig(dim=64, n_heads=4, n_kv_heads=2, dropout=0.0, max_seq_len=64)
    attention_model = Attention(args)

    B = 1
    L = 20
    x = torch.rand(B, L, args.dim)

    # 你之前有 precompute_rotary_freqs 或 precompute_freqs_cis，请确保导入了名称一致的函数
    freqs_cos, freqs_sin = precompute_rotary_freqs(args.dim // args.n_heads, L)

    # 正常前向（可能使用 flash），仅获得输出（不用于可视化）
    out = attention_model(x, freqs_cos, freqs_sin)

    # ---------- 单独计算 Q/K 并算权重（slow path，仅用于可视化） ----------
    with torch.no_grad():
        # 生成线性层输出并 reshape 为 [B, L, H, D]（与 forward 中相同的顺序）
        xq = attention_model.wq(x).reshape(B, L, attention_model.n_local_heads, attention_model.head_dim)
        xk = attention_model.wk(x).reshape(B, L, attention_model.n_local_kv_heads, attention_model.head_dim)

        # 应用 RoPE 与 KV 重复（与 forward 保持一致）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        xk = repeat_kv(xk, attention_model.n_rep)

        # 转置为 [B, H, L, D]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        # 可选 mask（若模型使用 slow path 时注册了 mask）
        mask = None
        if hasattr(attention_model, "mask"):
            # mask 形状为 [1,1,max_seq,max_seq]，裁剪到实际长度并广播
            mask = attention_model.mask[:, :, :L, :L]

        # 计算并得到 CPU 上的权重张量 [B, H, L, L]
        weights = compute_attn_weights_for_vis(xq, xk, mask=mask, head_dim=attention_model.head_dim)

    # ---------- 保存 / 绘图 ----------
    out_dir = Path("attn_figs")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_attention_matrix(weights, batch_idx=0, head_idx=0, fname=str(out_dir / "head0.png"))
    plot_avg_attention(weights, batch_idx=0, fname=str(out_dir / "avg_heads.png"))
    plot_attention_for_query_token(weights, query_pos=3, batch_idx=0, head_idx=0, fname=str(out_dir / "query3_head0.png"))

    print(f"Saved attention figures to {out_dir.resolve()}")

