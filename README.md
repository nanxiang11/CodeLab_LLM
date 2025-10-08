![87f48ba1bcb85dab3f287fa5e5d15371](https://github.com/user-attachments/assets/3570a363-8006-4e3b-a05c-9191fd91daca)

<h1 align="center">🦙 CodeLab-LLaMA2</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square">
  <img src="https://img.shields.io/badge/Model-LLaMA2-blue?style=flat-square">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
  <img src="https://img.shields.io/badge/Level-Advanced-orange?style=flat-square">
  <a href="https://swanlab.cn/@kmno4/Happy-LLM/overview"><img src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg" alt="SwanLab"></a>
</p>


<div align="center">

📖 [在线阅读地址](https://icnckt01te7y.feishu.cn/wiki/ZdP3wZQ3CiGKSkkS2NacAzhmnid?from=from_copylink)
### *从架构到实现，从预训练到应用的完整大模型解析之旅*
### *0.05B 大模型玩耍的飞起*
</div>



> 🌟 **CodeLab-LLaMA2** 是一个聚焦 **理解 LLaMA2 大语言模型的内部原理与工程实现** 的开源项目。
> 这里没有“复制粘贴”的黑盒实现，而是 **逐组件、逐细节** 地剖析每个设计背后的思考。
> 让你不止会“跑通模型”，更能真正理解 “为什么要这样设计”。

---

## 📘 项目简介

在大语言模型迅猛发展的今天，很多人能使用模型，却未必真正理解它的“灵魂”。
**CodeLab-LLaMA2** 希望成为一个桥梁，帮助你系统地掌握从**理论 → 实现 → 训练 → 应用**的完整流程。
该模型仅仅只有 **0.05B** 参数量大小，让**单卡3090 24G也可以完成从0-1纯手撕预训练，自己写一个训练器**。

本项目将详细讲解：

* LLaMA2 的 **整体架构与组件设计动机**
* 各模块的 **核心实现逻辑**
* 完整的 **预训练 + SFT + 推理** 实践流程
* 以及在工程层面如何高效、低成本地运行大模型。

---

## 🎯 项目目标

| 维度      | 目标                              |
| :------ | :------------------------------ |
| 🧠 理论理解 | 深入理解 Transformer 与 LLaMA2 的内部结构 |
| ⚙️ 工程实现 | 从零搭建 LLaMA2 模型代码，掌握模块间的依赖关系     |
| 🧮 训练实战 | 完整实现预训练、SFT、LoRA等训练流程    |
| 🚀 应用部署 | 支持模型加载、推理与 RAG / Agent 集成       |
| 🧩 拓展优化 | 探索结构改进与训练优化策略                   |

---

## 🌱 学习收获

* ✅ 理解 LLaMA2 的底层架构与创新点
* ✅ 掌握 Transformer 的内部机制（Attention、FeedForward、LayerNorm 等）
* ✅ 学会从零实现每个模块的完整代码
* ✅ 理解预训练目标、优化策略与分布式训练机制
* ✅ 掌握微调（SFT / LoRA）的工程细节
* ✅ 能独立运行推理与部署任务

---

## 📚 内容导航

| 章节                             | 核心主题                                     | 状态 |
| :----------------------------- | :--------------------------------------- | :- |
| 🧭 前言                          | 项目背景、学习路径与环境配置                               | ✅  |
| 🧩 第 1 章：整体架构                  | LLaMA2 结构概览、计算流与参数层次                     | ✅  |
| 💻 第 2 章：动手实现                  | 逐组件实现 LLaMA2 关键模块（附代码）                   | ✅  |
| 🔬 第 3 章：预训练流程                 | 数据准备、分布式训练、优化器、Loss 设计               | ✅ |
| 🧑‍🏫 第 4 章：SFT 微调             | 有监督微调、PEFT、LoRA 实践                                 | ✅ |
| ⚙️ 第 5 章：推理与应用                 | 加载模型、推理接口、RAG 应用                          | 🧩 |
| 🧠 第 6 章：优化与扩展                 | 模型压缩、量化、性能调优                              | 🧩 |


---

## 🧠 学习路径建议

```text
1️⃣ 理解整体 → 2️⃣ 拆解模块 → 3️⃣ 动手实现 → 4️⃣ 完整训练 → 5️⃣ 部署应用
```

> 🔍 **建议节奏**
>
> * 每章聚焦一个核心模块
> * 理论讲解 + 设计思考 + 对应代码实现
> * 最后整合形成完整模型
> * 边学边实现，不只是阅读

---



## ⚙️ 环境配置

```bash
git clone https://github.com/yourusername/CodeLab_llama2.git
cd CodeLab_llama2
conda create -n CodeLab_llama2 python=3.10
conda activate CodeLab_llama2
pip install -r requirements.txt
```

---

## 📚 参考与学习资料
* [DataWhale](https://github.com/datawhalechina) 
* [📄 LLaMA2 官方论文 (Meta AI)](https://arxiv.org/abs/2307.09288)
* [📄 Transformer: Attention is All You Need](https://arxiv.org/abs/1706.03762)
* [🤗 HuggingFace Transformers 文档](https://huggingface.co/docs/transformers/index)
---

## 💬 交流与贡献

📮 欢迎在 Issues 中讨论模型原理、实现细节或提出改进建议。
💡 如果你发现错误、性能优化点或新的实现方式，请提交 PR！

---

## 🌟 致谢

感谢：
* [Meta AI](https://ai.meta.com/) 开源的 LLaMA 系列模型
* [DataWhale](https://github.com/datawhalechina) 对 LLM 学习生态的推动
* 以及所有致力于开放大模型研究的开发者与学习者 🙌

---

> 🧭 **CodeLab-LLaMA2** — 不止复现，更要理解。
> 探索每一行代码背后的思想。

---

## 🪪 开源协议

本项目采用 **《知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议》**（CC BY-NC-SA 4.0）进行许可。  

> 允许在 **非商业用途** 下自由分享与改编，  
> 但必须 **署名原作者**，并以 **相同协议** 方式共享。  

📄 查看完整协议：  
👉 [https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-Hans](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-Hans)


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=nanxiang11/CodeLab_LLM&type=Date)](https://www.star-history.com/?utm_source=chatgpt.com#nanxiang11/CodeLab_LLM&Date)

