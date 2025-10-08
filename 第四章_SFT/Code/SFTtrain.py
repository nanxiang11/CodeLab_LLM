# -*- coding: utf-8 -*-
import os
import argparse
import time
import math

import torch
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from contextlib import nullcontext

from transformers import AutoTokenizer

from ModelConfig import ModelConfig
from LLaMA2 import Transformer
from SFTMapDataset import SFTMapDataset

import swanlab


# ------------------- 工具函数 -------------------
def Logger(msg: str, rank=0):
    """
    多卡训练中只在主卡打印日志
    rank=0 表示主进程
    """
    if rank == 0:
        print(msg)


def get_lr(it, total_iters, args):
    """
    计算学习率：
    - 线性预热阶段：从0线性增长到目标学习率
    - 余弦退火阶段：按余弦衰减到最小学习率
    - 超过总迭代次数：保持最小学习率
    """
    warmup_iters = args.warmup_iters
    min_lr = args.learning_rate / 10
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    elif it > total_iters:
        return min_lr
    else:
        decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate - min_lr)


def save_model(model, save_dir, step, lm_config, rank=0):
    """
    保存模型：
    - DDP 多卡训练时，只让主卡(rank=0)保存模型
    - 处理 DataParallel 或 DDP 包装的模型
    """
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        path = f"{save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step + 1}.pth"
        torch.save(state_dict, path)
        Logger(f"模型保存: {path}", rank)


# ------------------- 模型初始化 -------------------
def init_model(args, lm_config, rank, gpu_id):
    """
    初始化模型 + tokenizer + DDP 包装
    """

    def count_parameters(model):
        """计算可训练参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/StudyLLM/HappyLLM/Tokenizer/tokenizer_k')

    # 创建 Transformer 模型并移动到当前 GPU
    model = Transformer(lm_config).to(args.device)

    # 加载预训练权重
    ckp = '/root/StudyLLM/NX_LLM/第三章_预训练流程/ModelZoom/base-CodeLab-50M/pretrain_512_16_6144_step67200.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)



    # DDP 包装模型，每个进程只处理自己对应的 GPU
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu_id],
        output_device=gpu_id,
    )

    Logger(f'LLM总参数量: {count_parameters(model) / 1e6:.3f} M', rank)
    return model, tokenizer


# 训练单个epoch，其实对于大模型来说，基本都是只会训练一次，因为这个数据量实在是非常非常庞大
def train_epoch(epoch, model, train_loader, optimizer, scaler, args, lm_config, ctx, iter_per_epoch, rank):
    """
    训练一个 epoch
    - 支持梯度累积
    - 支持混合精度
    - 支持 DDP
    """

    # 记录开始时间
    start_time = time.time()

    for step, (X, Y, Attention_mask) in enumerate(train_loader):
        # 将数据迁移到显卡
        X, Y, Attention_mask = X.to(args.device), Y.to(args.device), Attention_mask.to(args.device)

        # 一个简易的学习率调度器
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 使用混合精度前向传播
        with ctx:
            out = model(X, Y)  # 模型前向
            # 除以梯度累积步数
            loss = out.last_loss / args.accumulation_steps
            # 展平 mask
            loss_mask_flat = Attention_mask.view(-1)
            # 忽略 padding
            loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()

        # 反向传播获取梯度
        scaler.scale(loss).backward()

        # 进行梯度更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 优化器更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # ------------------- 日志 -------------------
        if step % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            Logger(
                f"Rank[{rank}] Epoch[{epoch + 1}/{args.epochs}] "
                f"Step[{step}/{iter_per_epoch}] "
                f"Loss:{loss.item() * args.accumulation_steps:.4f} "
                f"LR:{lr:.7f} Elapsed:{elapsed_time:.1f}s",
                rank
            )
            if args.use_swanlab and rank == 0:
                swanlab.log({"loss": loss.item() * args.accumulation_steps, "lr": lr})

        # ------------------- 模型保存 -------------------
        if (step + 1) % args.save_interval == 0:
            save_model(model, args.save_dir, step, lm_config, rank)


# ------------------- 主函数 -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ------------------- 基础训练参数 -------------------
    parser.add_argument("--out_dir", type=str, default="../ModelZoom/SFT-CodeLab-50M", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="模型训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="训练的学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_swanlab", action="store_true", default=True, help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str,
                        default="/root/autodl-tmp/Dataset/BelleGroup/BelleGroup_sft.jsonl",
                        help="训练数据路径")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代次数")
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1200, help="模型保存间隔")
    # DDP 本地 rank
    parser.add_argument("--local_rank", type=int, default=0, help="DDP local rank")
    args = parser.parse_args()

    # ------------------- 初始化 DDP & 自动适配 GPU -------------------
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_id = local_rank
    torch.cuda.set_device(gpu_id)
    args.device = f'cuda:{gpu_id}'

    # 获取总 GPU 数量
    ngpus = torch.cuda.device_count()

    # 检查 local_rank 是否超出可用 GPU 范围
    if gpu_id >= ngpus:
        raise RuntimeError(f"local_rank {gpu_id} 超出可用 GPU 范围 (0-{ngpus - 1})")

    Logger(f"[Rank {args.local_rank}] 使用 GPU {gpu_id} / 总 GPU 数量 {ngpus}", args.local_rank)

    # ------------------- SwanLab -------------------
    if args.use_swanlab and args.local_rank == 0:
        swanlab.login(api_key="你的 API key")
        swanlab.init(project="CodeLab-LLM", experiment_name="SFT-50M", config=args)

    # ------------------- 模型配置 -------------------
    lm_config = ModelConfig(dim=512, n_layers=16)
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)

    # ------------------- 混合精度上下文 -------------------
    ctx = nullcontext() if "cpu" in args.device else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # ------------------- 模型 & 数据 -------------------
    model, tokenizer = init_model(args, lm_config, args.local_rank, gpu_id)
    train_ds = SFTMapDataset(args.data_path, tokenizer, max_length=max_seq_len)

    # DDP 必须使用 DistributedSampler
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ------------------- 优化器 & 混合精度 scaler -------------------
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ------------------- 开始训练 -------------------
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # DDP 每轮必须重置 epoch 保证 shuffle
        train_epoch(epoch, model, train_loader, optimizer, scaler, args, lm_config, ctx, iter_per_epoch,
                    args.local_rank)

    if args.local_rank == 0:
        Logger("训练完成！")
    dist.destroy_process_group()