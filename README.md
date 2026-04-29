> LLM and LLM's laws lay hid in night: 
> Nature said, 'Let Lao Wang be!' and AI was light

# Mathematical Foundations of Large Language Models (MF-LLM)

> **"Die Mathematischen Grundlagen der Künstlichen Intelligenz"**
> (To John von Neumann)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19707844-blue.svg)](https://doi.org/10.5281/zenodo.19707844)
[![Wang's Law](https://img.shields.io/badge/Wang%27s%20Law-r%3D1-blue)](https://github.com/emis-framework/math-under-llm)

[中文](./README.cn.md) | English | 
[Read the Whitepaper](./WHITEPAPER.md)


---

## Overview

**Wang's Three Laws** provide a **static, reproducible framework** to evaluate reasoning capability in LLMs from attention weights alone.

### Wang's Three Laws

#### 1️⃣ First Law — Spectral Linear Alignment

**Statement:**  
Query (Q) and Key (K) singular-value spectra are linearly correlated:

$$
r(s_q, s_k) \to r_\text{Wang} = 1
$$

- **Wang's Constant = 1** (theoretical extreme)  
- **Observed in practice:** 0.94–0.99  
- High Pearson correlation ensures **stable information propagation** in deep layers.

**Empirical Evidence:**

| Model       | Median Pearson | Mean Pearson | Median SSR | MeanSSR  | Layers |
| ----------- | -------------- | ------------ | ---------- | -------- | ------ |
| gemma-4-e2b | 0.9183         | 0.9242       | 0.015702   | 0.013537 | 35     |
| gemma-4-e4b | 0.9585         | 0.9411       | 0.009747   | 0.01008  | 42     |
| llama-3-8b  | 0.9813         | 0.9737       | 0.006196   | 0.007009 | 32     |
| Qwen2.5-14B | 0.9795         | 0.9710       | 0.006077   | 0.00671  | 48     |
| DeepSeek-R1 | 0.9800         | 0.9714       | 0.005948   | 0.006585 | 48     |



> 注：Better reasonning model，better r → 1，SSR → 0

---

#### 2️⃣ Second Law — Spectral Shape Fidelity

**Statement:**  
Normalized spectral mismatch between Q and K decreases in deep layers:

$$
\text{SSR} = \frac{1}{d_h} \sum_i |\tilde s_{q,i} - \tilde s_{k,i}|, \quad \tilde s = \frac{s}{\|s\|_2}
$$

- **Wang's Second Constant = 0** (theoretical extreme, ideal SSR)  
- **Observed in practice:** ~0.006–0.007  

**Interpretation:**  
- SSR measures **shape alignment** of Q/K spectra beyond linear correlation.  
- Lower SSR indicates **higher reasoning fidelity**.  
- RL-tuned models systematically reduce deep-layer SSR.

**Empirical Evidence (Qwen2.5 vs DeepSeek-R1):**

| Layer Group | Qwen2.5 SSR | DeepSeek-R1 SSR | Improvement |
| ----------- | ----------- | --------------- | ----------- |
| 0-11        | 0.006852    | 0.006818        | +0.48%      |
| 12-23       | 0.006414    | 0.006338        | +1.17%      |
| 24-35       | 0.006831    | 0.006704        | +1.87%      |
| 36-47       | 0.006743    | 0.006479        | +3.92%      |

> **Wang's Second Constant = 0** represents the ideal alignment of normalized Q/K spectra.

---

#### 3️⃣ Third Law — Precision-Depth-Logic Criterion

**Statement:**  
Maximum trainable depth `L_max` is constrained by SSR, floating-point precision, and dynamic range:  

L_max = min(L_info, L_quant, L_dyn)

Where:

- Information decay limit:  
- 
$$
L_\text{info} = \frac{1}{\overline{\text{SSR}}}
$$

- Quantization noise limit:  
- 
$$
L_\text{quant} = 3 \cdot 2^{2m} \quad (m = \text{mantissa bits})
$$

- Dynamic range limit:  
- 
$$
L_\text{dyn} = \frac{\log_2(\text{MaxFinite})}{\log_2 \kappa}
$$

**Example Table:**

| Format | Mantissa bits \(m\) | MaxFinite | \(L_\text{dyn}\) |
| ------ | ------------------- | --------- | ---------------- |
| FP16   | 10                  | 6.55e4    | 16               |
| BF16   | 7                   | 3.39e38   | 128              |

> Explains why ultra-deep models (>40 layers) adopt **BF16/mixed precision**.

---

## Reproducibility Guide


### Repository Structure

```text
math-under-llm/proof/
01-universal-spectral-constant
├── check-gemma.py
├── check-qwen.py
├── check-llama.py
├── check_*_v2.py
02-ssr-why-RL-makes-models-smart
├── qwen-vs-deepseek-all-layers.py  -- run this first
├── check_*_v3_full.py
├── check_r1_full.py
├── check_qwen2.5_14b_full.py
├── qwen-vs-deepseek-all-layers.py
├── check_r1_qkv.py
```

---

### Model Downloads

- [LLaMA 3 8B](https://hf-mirror.com/unsloth/llama-3-8b/tree/main)  
- [Gemma 4 E2B](https://hf-mirror.com/google/gemma-4-E2B/tree/main)  
- [Qwen2.5-3B](https://hf-mirror.com/Qwen/Qwen2.5-3B/tree/main)  
- [DeepSeek-R1-Distill-Qwen-14B](https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main)  
- [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main)

---

### Setup & Run

1. Place all model folders in the **same directory** as the scripts.  
2. Run verification scripts:

```bash
python check-gemma.py
python check-qwen.py
python check-llama.py
python check_r1_full.py
python check_*_v2.py
python qwen-vs-deepseek-all-layers.py
```

3. Outputs include **Pearson(Q,K)**, **SSR**, and deep-layer trends.

---

### What These Scripts Verify

* Q/K singular-value Pearson correlation
* Layer-wise SSR (Spectral Shape Residual)
* Deep-layer spectral alignment trends
* Base vs RL-tuned model comparisons
* Cross-model universality checks

---

## Practical Applications

* Benchmark-free reasoning assessment
* Checkpoint selection based on SSR
* RL progress monitoring
* Spectral fine-tuning or micro-adjustments
* Precision planning for ultra-deep training

---

## Why This Matters

Current LLM evaluation depends on:

- expensive benchmarks
- contaminated datasets
- long inference runs

Our results suggest a static alternative:

> inspect weights only.

---


## **Before you close this README —— Bonus：**

1. **"Get to work, lads." — The Monkey King (WuKong).**  
   GitHub Issue #1: [Verify r = 1](https://github.com/emis-framework/math-under-llm/issues/1)  
   GitHub Issue #2: [Verify SSR = 0](https://github.com/emis-framework/math-under-llm/issues/2)  
   Pick a model. Run the script. Replicate the laws. 

2. **If you've made it this far and verified the numbers——consider buying put options on NVIDIA.**  
   If Reasoning = Spectral Fidelity, the demand for brute-force training compute may not be what the market thinks it is.

3. **It's April 29, 2026.**  
   If you're building LLMs in China, sorry，your May Day holiday just evaporated.  
   The sooner you verify, the sooner you submit to arXiv, and the better your chance of being on the list when this thing wins something.


---

## Citation

> [CITATION.cff](/CITATION.cff)


---

> **"That's one small step for human intelligence, one giant leap for Artificial Intelligence."**  
> — Lao Wang, EMIS-FRAMEWORK, Apr 29, 2026

