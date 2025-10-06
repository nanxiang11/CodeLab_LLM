import torch
from typing import Tuple

# -----------------------------
# 1计算旋转频率
# -----------------------------
def precompute_rotary_freqs(dim: int, seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    dim: 每个head的维度
    seq_len: 序列长度
    theta: 基数，通常10000
    """
    # dim维度的一半，用于正余弦计算
    half_dim = dim // 2

    # 频率：1 / theta^(i/dim)
    freqs = 1.0 / (theta ** (torch.arange(half_dim).float() / half_dim))

    # 生成 seq_len x half_dim 的矩阵
    t = torch.arange(seq_len).unsqueeze(1)
    freqs = t * freqs.unsqueeze(0)

    # 计算正余弦
    return torch.cos(freqs), torch.sin(freqs)

# -----------------------------
# 2调整频率形状以广播
# -----------------------------
def reshape_for_broadcast(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    将 freqs 调整成可以广播到 x 的形状
    """
    shape = [1] * x.ndim
    shape[1] = freqs.shape[0]  # 序列长度
    shape[-1] = freqs.shape[1]  # head维度
    return freqs.view(shape)

# -----------------------------
# 3应用旋转嵌入
# -----------------------------
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xq, xk: [batch, seq_len, n_head, head_dim]
    cos, sin: [seq_len, head_dim//2]
    """
    # 先拆成实部和虚部
    xq_r, xq_i = xq.reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    # 调整频率形状以广播
    cos = reshape_for_broadcast(cos, xq_r)
    sin = reshape_for_broadcast(sin, xq_r)

    # 应用旋转公式
    xq_rot = torch.stack([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], dim=-1).flatten(-2)
    xk_rot = torch.stack([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], dim=-1).flatten(-2)

    return xq_rot.type_as(xq), xk_rot.type_as(xk)

# -----------------------------
# 4示例
# -----------------------------
if __name__ == "__main__":
    batch, seq_len, n_head, head_dim = 1, 50, 6, 48
    xq = torch.randn(batch, seq_len, n_head, head_dim)
    xk = torch.randn(batch, seq_len, n_head, head_dim)

    cos, sin = precompute_rotary_freqs(head_dim, seq_len)
    xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)

    print("xq_out shape:", xq_out.shape)
    print("xk_out shape:", xk_out.shape)
