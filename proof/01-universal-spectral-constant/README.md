check-llama-v2.py 谱分析程序编写的 `README.md` 文件，包含详细的背景、使用方法、Pearson 与 Spearman 相关系数的数学解释及实践解读。


# 注意力头谱形状一致性分析工具 (Spectral Shape Consistency Analyzer)

本工具用于验证 **GQA (Grouped Query Attention)** 架构中，同一组内的 Query 头与 Key 头的奇异值谱形状是否满足理论预期的线性或单调相关性。通过逐层、逐头组计算 **Pearson 相关系数** 与 **Spearman 秩相关系数**，帮助研究者判断模型是否自发学习到了“逻辑等距映射”性质。

## 🔬 背景

在 Transformer 的 GQA 设计中，多个 Query 头共享一个 Key/Value 头。理论分析指出：若 Q 与 K 的投影矩阵满足谱半径 ρ≈1 且奇异值谱形状高度相似（或呈现反比关系），则注意力分数计算将具有更好的数值稳定性和泛化能力。

本程序从已训练模型的 `safetensors` 权重文件中提取 Q 与 K 矩阵，执行以下验证：

- 对每个头的权重矩阵进行奇异值分解 (SVD)
- 比较同一 KV 头与组内每个 Q 头的奇异值序列
- 输出两种相关系数，综合判断谱形状的匹配程度

## 📦 依赖安装

```bash
pip install torch safetensors numpy scipy
```

## 🚀 使用方法

### 1. 准备模型权重文件

模型下载link，可自行下载模型后，跑python 脚本验证：（可以之下第一个分片）

https://hf-mirror.com/unsloth/llama-3-8b/tree/main

https://hf-mirror.com/google/gemma-4-E2B/tree/main

https://hf-mirror.com/Qwen/Qwen2.5-3B/tree/main



将待分析的 GQA 模型权重保存为 `.safetensors` 格式（例如 `model-00001-of-00004.safetensors`），并确保其包含 Q 与 K 权重张量，命名格式如：
- `model.layers.{L}.self_attn.q_proj.weight`
- `model.layers.{L}.self_attn.k_proj.weight`

### 2. 修改配置参数

打开脚本，根据你的模型调整以下超参数：（gemma-4的最复杂，直接看code吧）

```python
N_Q_HEADS = 32      # Query 头总数
N_KV_HEADS = 8      # Key/Value 头总数
D_MODEL = 4096      # 隐藏层维度
D_HEAD = 128        # 每个头的维度
LAYERS_TO_ANALYZE = range(32)   # 待分析的层索引（如 [0,1,2] 或 range(32)）
FILE_PATH = "model-00001-of-00004.safetensors"   # 权重文件路径
```

### 3. 运行脚本

```bash
python check-llama-v2.py
```

输出示例：
```
========== Layer 0 ==========
  KV头 0 (组内Q偏移 0) | Q头索引  0: Pearson = +0.9431, Spearman = +1.0000
  KV头 0 (组内Q偏移 1) | Q头索引  1: Pearson = +0.9872, Spearman = +1.0000
  ...
```

## 📊 输出指标解读

### Pearson 相关系数 (r)

- **数学本质**：衡量两个变量之间的 **线性关系强度**。  
- **公式**：  
  \[
  r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}
  \]
- **取值范围**：[-1, 1]  
  - `+1`：完全正线性相关（`y = a + bx, b>0`）  
  - `-1`：完全负线性相关（`y = a + bx, b<0`）  
  - `0`：无线性关系  

**在本程序中的意义**：  
若 Q 头的奇异值 \(s_q\) 与对应 KV 头的奇异值 \(s_k\) 满足近似线性关系（例如 \(s_q \approx \alpha s_k + \beta\)），则 Pearson 系数接近 +1。如果线性关系被破坏（如出现极端奇异值），即使单调性良好，Pearson 也会降低。

### Spearman 秩相关系数 (ρ)

- **数学本质**：衡量两个变量之间的 **单调关系强度**（不要求线性）。  
- **计算过程**：  
  1. 将 \(x\) 和 \(y\) 分别排序，得到秩次 \(R(x_i)\)、\(R(y_i)\)  
  2. 计算秩次的 Pearson 相关系数  
- **取值范围**：[-1, 1]  
  - `+1`：完全单调递增（\(x\) 增加时 \(y\) 总是增加）  
  - `-1`：完全单调递减（\(x\) 增加时 \(y\) 总是减少）

**在本程序中的意义**：  
即使 \(s_q\) 与 \(s_k\) 之间是非线性但严格单调的关系（例如 \(s_q = a / s_k\) 或 \(s_q = a \cdot e^{b s_k}\)），Spearman 仍会接近 ±1。因此当 Pearson 偏低但 Spearman 很高时，说明谱形状呈现完美的单调关系，可能符合某种理论预期（如反比律）。

## 🧠 为什么使用两种指标？

| 场景         | Pearson  | Spearman | 结论                                   |
| ------------ | -------- | -------- | -------------------------------------- |
| 线性关系良好 | 接近 +1  | 接近 +1  | Q 与 K 奇异值呈**线性正相关**          |
| 单调但非线性 | 明显偏低 | 接近 +1  | 谱形状匹配但需要非线性变换（如取倒数） |
| 无单调性     | 接近 0   | 接近 0   | 谱形状无显著关联                       |
| 完全负单调   | 接近 -1  | 接近 -1  | 奇异值呈反比关系（可能理论预期）       |

**常见情况**：  
- 理论预期是 \(s_q \approx s_k\)（正比关系），则 **Spearman = 1.0** 比 Pearson 更能验证单调性。  
- 验证线性关系（如正则化后的等距映射），则 **Pearson 应接近 +1**。

## ⚙️ 硬件加速

脚本自动检测 CUDA 环境：  
- 所有 SVD、张量运算在 GPU 上执行（使用 `torch.linalg.svd`）  
- Pearson 相关系数也在 GPU 上计算，避免 CPU-GPU 频繁拷贝  
- Spearman 计算需要排序，数据暂时移至 CPU（使用 SciPy），但数据量极小（仅奇异值向量，长度 = `head_dim`），不影响性能

## 📝 注意事项

1. **奇异值截断**：Q 头和 KV 头的矩阵形状可能不完全相同（例如 Q 头参与更多头数），奇异值向量长度可能不同。程序会自动截取较短长度，保留前 `min(len(s_q), len(s_k))` 个奇异值（因奇异值降序排列，头部含有主要信息）。

2. **权重加载**：脚本通过自动识别第 4 层的 Q 权重键名生成模板。若你的模型命名规则不同（如使用 `wq` 而非 `q_proj`），请手动修改 `template_q` 和 `template_k` 的正则匹配规则。

3. **数值稳定性**：奇异值可能极小，计算 Pearson 时加法与除法会引入微小误差，但对整体相关系数影响可忽略（< 1e-5）。


## 🤝 贡献与反馈

如有问题或改进建议，欢迎提交 Issue 或 Pull Request。

---

**作者声明**：本工具用于学术研究中的谱分析验证，不承担模型训练或商业用途责任。
