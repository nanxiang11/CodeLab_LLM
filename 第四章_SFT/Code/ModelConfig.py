from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"

    def __init__(self,
                 dim: int = 512,           # 模型维度
                 n_layers: int = 16,       # Transformer层数
                 n_heads: int = 16,        # 多头
                 n_kv_heads: int = 8,      # kv数量
                 vocab_size: int = 6144,   # 词汇表大小
                 hidden_dim: int = None,   # 隐藏层大小
                 multiple_of: int = 64,    # 隐藏层对齐数
                 norm_eps: float = 1e-5,   # 归一化层的平滑项
                 max_seq_len: int = 256,   # 最大序列长度
                 dropout: float = 0.0,
                 flash_attn: bool = True,  # 是否使用Flash Attention
                 **kwargs,
                 ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)