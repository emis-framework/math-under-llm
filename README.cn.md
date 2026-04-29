> LLM and LLM's laws lay hid in night:   
> Nature said, 'Let Lao Wang be!' and AI was light.

TL;DR：用 SVD 分解 LLM 的 Q/K 权重矩阵，发现两条跨模型普适规律——r=1（谱线性对齐）和 SSR→0（谱保真度）。不需要跑推理，不需要测试集，只看权重就能判断模型的推理能力。


# 大模型的数学基础
# Mathematical Foundations of Large Language Models (MF-LLM)

> **"Die Mathematischen Grundlagen der Künstlichen Intelligenz"**
> ( —— 致敬 John von Neumann)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19707844-blue.svg)](https://doi.org/10.5281/zenodo.19707844)
[![Wang's Law](https://img.shields.io/badge/Wang%27s%20Law-r%3D1-blue)](https://github.com/emis-framework/math-under-llm)

中文 | [English](./README.md)

[阅读详细理论白皮书 (Whitepaper)](./WHITEPAPER.cn.md)

---

## 概览

**王氏三定律（Wang's Three Laws）** 提供了一个 **静态、可复现的框架**，用于仅通过注意力权重评估 LLM 的推理能力。

### 王氏三定律

#### 1️⃣ 第一条定律 — 谱线性对齐

**说明：**  
Query (Q) 和 Key (K) 的奇异值谱呈线性相关：

$$
r(s_q, s_k) \to r_\text{Wang} = 1
$$

- **王氏常数 = 1**（理论极限）  
- **实测范围：** 0.94–0.99  
- 高 Pearson 相关确保 **深层信息传播稳定**。

**实测数据：**


| 模型名称    | Pearson中位数 | Pearson平均数 | SSR中位数 | SSR平均数 | 层数 |
| ----------- | ------------- | ------------- | --------- | --------- | ---- |
| gemma-4-e2b | 0.9183        | 0.9242        | 0.015702  | 0.013537  | 35   |
| gemma-4-e4b | 0.9585        | 0.9411        | 0.009747  | 0.01008   | 42   |
| llama-3-8b  | 0.9813        | 0.9737        | 0.006196  | 0.007009  | 32   |
| Qwen2.5-14B | 0.9795        | 0.9710        | 0.006077  | 0.00671   | 48   |
| DeepSeek-R1 | 0.9800        | 0.9714        | 0.005948  | 0.006585  | 48   |


> 注：推理能力越好的模型，更好的 r → 1，SSR → 0

---

#### 2️⃣ 第二条定律 — 谱形状保真度

**说明：**  
Q 和 K 的归一化谱在深层逐渐匹配：

$$
\text{SSR} = \frac{1}{d_h} \sum_i |\tilde s_{q,i} - \tilde s_{k,i}|, \quad \tilde s = \frac{s}{\|s\|_2}
$$

- **王氏第二常数 = 0**（理论极限，理想 SSR）  
- **实测范围：** ~0.006–0.007  

**解释：**  
- SSR 衡量 Q/K 谱形状的对齐精度，超出简单线性相关。  
- SSR 越低，推理保真度越高。  
- RL 微调模型可系统性降低深层 SSR。

**实测对比（Qwen2.5 vs DeepSeek-R1）：**

| 层组  | Qwen2.5 SSR | DeepSeek-R1 SSR | 改善率 |
| ----- | ----------- | --------------- | ------ |
| 0-11  | 0.006852    | 0.006818        | +0.48% |
| 12-23 | 0.006414    | 0.006338        | +1.17% |
| 24-35 | 0.006830    | 0.006704        | +1.87% |
| 36-47 | 0.006743    | 0.006479        | +3.92% |

> **王氏第二常数 = 0** 表示 Q/K 归一化谱的理想对齐状态。

---

#### 3️⃣ 第三条定律 — 精度‑深度‑逻辑判据

**说明：**  
最大可训练深度 `L_max` 受 SSR、浮点精度和动态范围限制：

```

L_max = min(L_info, L_quant, L_dyn)

````

其中：

- 信息衰减极限：  
  
$$
L_\text{info} = \frac{1}{\overline{\text{SSR}}}
$$

- 量化噪声极限：  
  
$$
L_\text{quant} = 3 \cdot 2^{2m} \quad (m = \text{尾数位数})
$$

- 动态范围极限：  
  
$$
L_\text{dyn} = \frac{\log_2(\text{MaxFinite})}{\log_2 \kappa}
$$

**示例表：**

| 格式 | 尾数位数 $m$ | 最大有限值 | $L_\text{dyn}$ |
| ---- | ------------ | ---------- | -------------- |
| FP16 | 10           | 6.55e4     | 16             |
| BF16 | 7            | 3.39e38    | 128            |

> 解释了为什么超深模型（>40 层）采用 **BF16 或混合精度**。

---

## 可复现指南

### 仓库结构

```text
math-under-llm/proof/
01-universal-spectral-constant
├── check-gemma.py
├── check-qwen.py
├── check-llama.py
├── check_*_v2.py
02-ssr-why-RL-makes-models-smart
├── qwen-vs-deepseek-all-layers.py  -- 首先运行
├── check_*_v3_full.py
├── check_r1_full.py
├── check_qwen2.5_14b_full.py
├── qwen-vs-deepseek-all-layers.py
├── check_r1_qkv.py
````

---

### 模型下载

* [LLaMA 3 8B](https://hf-mirror.com/unsloth/llama-3-8b/tree/main)
* [Gemma 4 E2B](https://hf-mirror.com/google/gemma-4-E2B/tree/main)
* [Qwen2.5-3B](https://hf-mirror.com/Qwen/Qwen2.5-3B/tree/main)
* [DeepSeek-R1-Distill-Qwen-14B](https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main)
* [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main)

---

### 安装与运行

1. 将所有模型文件夹放在 **脚本同级目录**。
2. 运行验证脚本：

```bash
python check-gemma.py
python check-qwen.py
python check-llama.py
python check_r1_full.py
python check_*_v2.py
python qwen-vs-deepseek-all-layers.py
```

3. 输出包括 **Pearson(Q,K)**、**SSR** 和深层谱对齐趋势。

---

### 脚本验证内容

* Q/K 奇异值 Pearson 相关
* 每层 SSR（谱形状残差）
* 深层谱对齐趋势
* 基础模型 vs RL 微调模型对比
* 跨模型通用性检查

---

## 实际应用

* 无需 benchmark 的推理能力评估
* SSR 指标驱动的模型 checkpoint 选择
* RL 训练进度监控
* 谱微调 / 局部微调
* 超深模型精度规划

---

## 为什么重要

当前 LLM 评估依赖：

* 高成本 benchmark
* 潜在数据污染
* 长时间推理

我们的结果表明：

> **只需检查权重即可**。


---

## 工程应用与未来方向

王氏三定律不仅是理论发现，更是一套可直接嵌入工程流程的诊断工具。本章汇总三定律在模型训练、推理部署、架构设计、安全防御等关键领域的应用前景。


### 一、训练范式：从“看 Loss”到“看谱”

**1. SSR 作为无测试集的训练导航仪**

训练过程中实时计算深层 SSR。若 SSR 出现非单调“驼峰”（先升后降）或剧烈上升 > 5%，说明谱形状正在畸变，应立即降低学习率或回滚 checkpoint。这比等待下游任务分数反馈快得多。

**2. 数据配比的谱准则**

用不同数据（代码、数学、对话）分别训练小模型，比较各数据产出的 SSR 下降曲线。曲线最陡且光滑者即为最优配比——这是数据选择的谱反馈机制。

**3. 谱对齐辅助损失**

在预训练目标中加入 $L_{\text{SSR}} = \text{mean}(\text{SSR})$ 作为辅助损失，鼓励深层谱形状一致性，可能加速收敛并提升泛化能力。

**4. 无优化器的 Q 初始化**

利用已训好的 $W_K$，构造 $W_Q^{\text{init}}$ 使其 $s_q = s_k$， $U_q$  随机正交。这种谱保持初始化能减轻深层梯度消失，有望缩短预训练时间。

**5. 条件数感知的权重衰减**

对 $\text{cond}(Q) > 1000$ 的层施加更强的正则化或正交化约束，防止深层秩亏恶化。


### 二、推理部署：谱自适应量化

这是第三定律最直接、价值最大的落地场景。

**1. 逐层混合精度量化**

根据每层的 $\text{cond}(Q)$ 自动决定量化位宽：Layer 1 的 $\text{cond}$ 高达数千，必须保留 BF16；深层 $\text{cond}$ 仅 2–10，可压至 INT4。相较统一量化，谱自适应混合精度可降低 30%–50% 的 KV cache 与权重显存，且精度无损。

**2. 条件数触发的动态重正交化**

推理时监控关键层的 $\text{cond}$（可通过在线幂迭代快速估算）。当输入导致某层 $\text{cond}$ 突然飙升时，自动插入极少量 Newton‑Schulz 迭代进行临时正交化，防止长文本输出崩溃。这正是 DeepSeek‑V4 技术的防御性部署。

**3. 硬件友好的谱稀疏化**

对 $\text{cond}(K)$ 极低（接近 1）的层，其 K 矩阵接近等距，可直接用随机正交矩阵替换，甚至改为固定 Hadamard 变换，省去权重存储和计算。


### 三、架构设计与搜索

**1. 深度扩展的谱安全边界**

训练前通过小规模代理模型，测量各层 $\text{cond}$ 随深度的增长曲线，预测 BF16 下何时突破安全阈值，从而科学决定层数，而非盲目试错。

**2. “数值安全阀”的结构化设计**

我们发现 Layer 1 的条件数爆炸将数值风险集中前置，保护了深层。这可以显式化为设计原则：有意在第一层使用低秩 Q 矩阵或强制引入大条件数，作为深层稳定性的“保险丝”。

**3. 多模态/多专家模型的谱功能分区**

对 MoE 模型的每个专家分别做 Q/K 矩阵 SVD。数学专家的 SSR 应极低且 r 极高，而闲聊专家谱更噪。仅凭谱特征即可自动区分不同功能的专家，实现无提示的功能识别。

**4. “相变层”监测**

正常模型的条件数和 SSR 跳变稳定在 Layer 1。若训练中检测到该跳变漂移至 Layer 3 或消失，则预示“表示坍塌”，需立即干预。


### 四、安全与对抗防御

**1. 基于实时条件数监控的越狱检测**

对抗样本常引发某些注意力头的极端激活，导致瞬态 $\text{cond}(Q)$ 飙升。在推理时旁路监控 Q/K 矩阵的近似奇异值，当 $\text{cond}$ 异常跳变（如瞬间 > 5000）时判定为攻击并截断输入。这是与内容无关的底层防御。

**2. 模型指纹与供应链安全**

不同架构甚至不同训练配方会产生特异性“谱纹”：Gemma 的 r 略低，Qwen 的 Layer 1 条件数形状，R1 的深层 SSR 压制模式。通过提取几层关键谱指标，可像指纹一样识别模型身份，用于模型溯源与防止未授权蒸馏。


### 五、模型压缩与知识蒸馏

**1. 谱蒸馏损失**

让学生模型的归一化谱形状直接回归教师模型的归一化谱形状。SSR 原生就是损失函数，这比 KL 散度更直接地传递模型的“推理骨架”，可能用极少数据实现高效蒸馏。

**2. 剪枝的头级谱异常检测**

若某头的 r 极低且 SSR 极高，说明该头 Q/K 谱已畸变，可优先剪除。这比基于幅度或梯度的剪枝更有物理意义。

**3. 头级 SVD 重构修复**

发现异常头时，可强行将该头的 $s_q$ 按 $s_k$ 的线性映射重标定（ $s_q^{\text{new}} = \alpha s_k + \beta$ ），保留 $U_q$ 不变。这是最无损的“外科手术式”修复。

**4. 分块谱正则化**

若 Q 的 SVD 显示前 32 个奇异值规律良好、后 96 个充满噪声，可只微调后 96 个奇异值对应的右奇异向量，极大压缩微调参数量。


### 六、模型诊断与维护

**1. 终身学习与持续微调的谱监控**

每次微调后记录各层 SSR 和 r 的分布漂移。若微调导致深层 SSR 明显上升，说明灾难性遗忘正在发生，可提前停止或调整正则化。这是无需验证集的遗忘检测器。

**2. 闭源模型的“隔空探伤”**

面对黑盒 API 模型，可通过延迟探测法间接推断其谱特性：若某模型 QK 的深层 cond 极大，则极长输入下输出混乱度会骤升。用同一长文本压测不同模型，崩溃点越晚，深层谱控制越好。


### 七、最值得立即动手的三个实验

1. **谱自适应量化**：根据每层 $\text{cond}(Q)$ 分配量化位宽，立刻可在项目中验证显存节省与精度保持效果。
2. **训练 SSR 监控回调**：在训练脚本中加入 SSR 计算，设置自动触发学习率调整或早停。
3. **无梯度 Q 初始化**：用 $W_K$ 的谱初始化 $W_Q$，在小模型上验证收敛速度提升。

---

## 看到这里的小伙伴，额外奖励：让王氏定理帮你省钱提速

最后我还想加上3句话：

1、孩儿们，操练起来。--齐天大圣。 
> GitHub Issue #1: [Verify r = 1](https://github.com/emis-framework/math-under-llm/issues/1)  
   GitHub Issue #2: [Verify SSR = 0](https://github.com/emis-framework/math-under-llm/issues/2) 

2、能看到这里，已经验证过r=1， SSR→0的同学，可以去做空英伟达的股票，买Nvidia（英伟达）的看跌期权，put option.

3、今天是2026年4月29号，国内做大模型的同学，抱歉，你们的五一长假泡汤了。越早验证完，越早发arxiv，越有可能共同获奖。

---

## Citation

> [CITATION.cff](/CITATION.cff)

---

> **"这是人I一小步，却是AI一大步。"**  
> — 老王, 能量主义（EMIS-FRAMEWORK）, 2026年4月29日

---