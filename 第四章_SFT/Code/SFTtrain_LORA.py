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
from SFTDataset import SFTDataset

from peft import LoraConfig, get_peft_model, TaskType
import swanlab

# ------------------- 工具函数 -------------------
def Logger(msg: str, rank=0):
    if rank == 0:
        print(msg)

def get_lr(it, total_iters, args):
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
    保存 LoRA 微调模型（兼容 DDP 和多卡）
    文件名和原始预训练保存一致：
    pretrain_{dim}_{n_layers}_{vocab_size}_step{step}.pth

    Args:
        model: 当前模型，可能是 DDP 包装的
        save_dir: 保存目录
        step: 当前训练步数
        lm_config: 模型配置（dim, n_layers, vocab_size）
        rank: 当前进程 rank，只让主卡保存
    """
    if rank != 0:
        return  # 只有主卡保存

    os.makedirs(save_dir, exist_ok=True)

    # 如果是 DDP 包装的模型，需要取 module
    model_to_save = model.module if hasattr(model, 'module') else model

    # 构建完整的保存路径
    path = os.path.join(
        save_dir,
        f"pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step + 1}"
    )

    # 使用 HuggingFace 的 save_pretrained 保存 LoRA 权重
    model_to_save.save_pretrained(path)

    Logger(f"LoRA 模型已保存: {path}", rank)


# ------------------- 模型初始化 -------------------
def init_model(args, lm_config, rank, gpu_id):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained('/root/StudyLLM/HappyLLM/Tokenizer/tokenizer_k')
    model = Transformer(lm_config).to(args.device)

    # 加载原始预训练权重
    ckp = '/root/StudyLLM/NX_LLM/第三章_预训练流程/ModelZoom/base-CodeLab-50M/pretrain_512_16_6144_step67200.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)

    # ------------------- LoRA 包装 -------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["wq", "wv", "output"],  # 根据 Transformer 实现选择
    )
    model = get_peft_model(model, lora_config)

    # DDP 包装
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu_id],
        output_device=gpu_id,
    )

    Logger(f'LLM 可训练参数量: {count_parameters(model)/1e6:.3f} M', rank)
    return model, tokenizer

# ------------------- 训练 -------------------
def train_epoch(epoch, model, train_loader, optimizer, scaler, args, lm_config, ctx, iter_per_epoch, rank):
    start_time = time.time()
    for step, (X, Y, Attention_mask) in enumerate(train_loader):
        X, Y, Attention_mask = X.to(args.device), Y.to(args.device), Attention_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask_flat = Attention_mask.view(-1)
            loss = torch.sum(loss * loss_mask_flat) / loss_mask_flat.sum()

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            Logger(
                f"Rank[{rank}] Epoch[{epoch+1}/{args.epochs}] "
                f"Step[{step}/{iter_per_epoch}] "
                f"Loss:{loss.item()*args.accumulation_steps:.4f} "
                f"LR:{lr:.7f} Elapsed:{elapsed_time:.1f}s",
                rank
            )
            if args.use_swanlab and rank == 0:
                swanlab.log({"loss": loss.item()*args.accumulation_steps, "lr": lr})

        if (step + 1) % args.save_interval == 0:
            save_model(model, args.save_dir, step, lm_config, rank)

# ------------------- 主函数 -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="../ModelZoom/SFT-CodeLab-50M-LORA")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_swanlab", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/Dataset/BelleGroup/BelleGroup_sft.jsonl")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1200)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # DDP 初始化
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_id = local_rank
    torch.cuda.set_device(gpu_id)
    args.device = f'cuda:{gpu_id}'

    ngpus = torch.cuda.device_count()
    if gpu_id >= ngpus:
        raise RuntimeError(f"local_rank {gpu_id} 超出可用 GPU 范围 (0-{ngpus-1})")
    Logger(f"[Rank {args.local_rank}] 使用 GPU {gpu_id} / 总 GPU 数量 {ngpus}", args.local_rank)

    if args.use_swanlab and args.local_rank == 0:
        swanlab.login(api_key="你的 API KEY")
        swanlab.init(project="CodeLab-LLM", experiment_name="SFT-50M-LORA", config=args)

    lm_config = ModelConfig(dim=512, n_layers=16)
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)

    ctx = nullcontext() if "cpu" in args.device else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    model, tokenizer = init_model(args, lm_config, args.local_rank, gpu_id)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # 只训练 LoRA 参数
        lr=args.learning_rate
    )

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_epoch(epoch, model, train_loader, optimizer, scaler, args, lm_config, ctx, iter_per_epoch, args.local_rank)

    if args.local_rank == 0:
        Logger("LoRA 微调完成！")
    dist.destroy_process_group()
