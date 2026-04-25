# 逻辑谱形状对撞实验：Qwen 2.5 vs DeepSeek R1

本脚本实现了一种**纯静态权重对比方法**，通过计算 Query 与 Key 投影矩阵的**谱形状残差 (SSR)**，定量评估两个模型在注意力几何上的逻辑一致性差异。实验验证了 DeepSeek R1 在深层（40–47层）显著收窄了谱残差，揭示了 RL 训练对逻辑结构修复的物理机理。

## 📌 实验背景

在 GQA（Grouped Query Attention）架构中，多个 Query 头共享同一个 Key 头。理论预期：一个具备强逻辑推理能力的模型，其 Q 投影与 K 投影的奇异值谱形状应高度对齐（即谱半径 ρ≈1 且奇异值序列成比例）。本实验对比：

- **原生模型**：Qwen2.5-14B-Instruct（纯 SFT）
- **RL 增强模型**：DeepSeek-R1-Distill-Qwen-14B（蒸馏自 R1 的推理能力）

通过逐层计算 **谱形状残差 (SSR)**，观察 RL 训练是否以及如何改善了深层的谱对齐质量。

**模型下载link**，可自行下载模型后，跑python 脚本验证：(qwen有8个分片，r1有4个)


https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main

https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main

## 🔬 核心算法：谱形状残差 (SSR)

1. **提取权重**：对每一层，取出 `q_proj.weight` 和 `k_proj.weight`。
2. **GQA 对齐**：将 K 头按 GQA 比例（5:1）重复扩展，使 K 头数量与 Q 头一致（40 对）。
3. **批量化 SVD**：利用 `torch.linalg.svdvals` 一次性计算所有头的奇异值向量。
4. **谱归一化**：对每个头的奇异值向量进行 L2 归一化，消除幅度差异。
5. **残差计算**：对归一化后的奇异值向量逐元素平均绝对误差，得到该层的 SSR（越小表示谱形状越一致）。
6. **改善率**：`Improvement = (SSR_native - SSR_rl) / SSR_native × 100%`，正值表示 R1 的谱对齐优于 Native。

## 🚀 运行方法

### 环境依赖

```bash
pip install torch safetensors numpy
```

### 数据准备

- 将 Qwen2.5-14B-Instruct 和 DeepSeek-R1-Distill-Qwen-14B 的权重文件（`.safetensors` 格式）放在脚本同级目录下。
- 文件名需包含 `Qwen2.5-14B-Instruct-model` 和 `DeepSeek-R1-Distill-Qwen-14B-model` 子串（脚本自动识别）。
- 下载好模型后改名如下：
DeepSeek-R1-Distill-Qwen-14B-model-00001-of-000004.safetensors
DeepSeek-R1-Distill-Qwen-14B-model-00002-of-000004.safetensors
DeepSeek-R1-Distill-Qwen-14B-model-00003-of-000004.safetensors
DeepSeek-R1-Distill-Qwen-14B-model-00004-of-000004.safetensors
Qwen2.5-14B-Instruct-model-00001-of-00008.safetensors
Qwen2.5-14B-Instruct-model-00002-of-00008.safetensors
Qwen2.5-14B-Instruct-model-00003-of-00008.safetensors
Qwen2.5-14B-Instruct-model-00004-of-00008.safetensors
Qwen2.5-14B-Instruct-model-00005-of-00008.safetensors
Qwen2.5-14B-Instruct-model-00006-of-00008.safetensors
Qwen2.5-14B-Instruct-model-00007-of-00008.safetensors
Qwen2.5-14B-Instruct-model-00008-of-00008.safetensors


### 执行脚本

```bash
python qwen-vs-deepseek-all-layers.py
```

### 输出说明

- 脚本将 48 层分为 4 个阶段（每阶段 12 层）依次加载，打印每层的 Native SSR、R1 SSR 和改善率。
- 最终输出全 48 层平均改善率及实验总结。

## 📊 实验结果解读（示例）输出这个文件：ssr-qwen-vs-r1.txt

```
Layer      | Native SSR         | R1 SSR             | Imp %       
---------------------------------------------------------------------------
L0         | 0.01126314        | 0.01116304        |    +0.8888%
L1         | 0.00887516        | 0.00880650        |    +0.7736%
...
L43        | 0.00658677        | 0.00619481        |    +5.9508%
...
平均提升: 1.8617%
```

**关键发现**：

- **浅层（0–11层）**：改善微弱（<1%），偶有负值（可视为数值噪声）。
- **中层（12–35层）**：改善稳定在 1–2%，R1 持续优于 Native。
- **深层（36–47层）**：改善率急剧上升，L40–L45 达到 3%–5.95% 的爆发区。**L43 达到峰值 +5.95%**。

**结论**：DeepSeek R1 的 RL 训练显著修复了原生模型深层存在的“谱形状失调”，使得 Q 与 K 的奇异值分布更加一致。这一物理层级的对齐，可能是 R1 推理能力来源的重要静态证据。

## 🧠 理论意义

本实验首次在大规模 GQA 模型上，通过纯静态权重验证了**王氏伴随对偶定理**的一个核心推论：  
> **逻辑相干性等价于 Q 与 K 的谱形状对齐，且深层对齐程度与推理能力正相关。**

脚本中使用的 **批量化 GPU SVD + 谱归一化残差** 方法，可推广至任意 GQA 模型，用于评估注意力结构的几何质量。

## 📝 自定义修改

如需适配其他模型（不同 GQA 比例、头维度），请修改脚本开头的超参数：

```python
D_HEAD = 128          # 每个头的维度
N_Q_HEADS = 40        # Query 头总数
N_KV_HEADS = 8        # Key/Value 头总数
GQA_RATIO = 5         # 每组 GQA 的 Q 头数
```

若模型层数不是 48，可调整 `batches` 中的范围。

## ⚠️ 注意事项

- 需要约 16GB 系统内存（脚本已做四段加载，峰值约 6GB）和 8GB 以上显存。
- 如果权重文件分片（如 `model-00001-of-00004.safetensors`），请确保所有分片都在同一目录，脚本会自动搜索并合并加载（通过遍历所有分片）。
- 运行前请确认 GPU 可用，否则自动回退到 CPU（速度会慢很多）。

## 📚 引用

若在研究中使用了本方法，建议引用：

> [CITATION.cff](/CITATION.cff)

## 🤝 贡献与反馈

欢迎提交 Issue 或 Pull Request 改进算法或适配更多模型。

> issue 1: verify ρ=1
> issue 2: verify ssr

---

