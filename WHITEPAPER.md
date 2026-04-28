# Wang's Three Laws: Spectral Signals Inside Large Language Model Attention

**Author:** Wang Fei-Yun  
**Date:** 2026-04-28  
**Version:** v4.0  
**DOI:** 10.5281/zenodo.19707844  
**GitHub:** [math-under-llm](https://github.com/emis-framework/math-under-llm)

---

## Executive Summary

Reasoning in LLMs can be inferred not only via benchmarks but also directly from **attention projection weights**. Across multiple open-source LLM families, we formalize **Wang's Three Laws**:

1. **First Law — Spectral Linear Alignment:** Q/K singular-value spectra are linearly correlated. The **theoretical extreme**, **Wang's Constant**, is 1.  
2. **Second Law — Spectral Shape Fidelity:** Normalized spectral mismatch (SSR) decreases in deep layers. The **ideal SSR**, **Wang's Second Constant**, is 0.  
3. **Third Law — Precision-Depth-Logic Criterion:** Maximum trainable depth is constrained by SSR, floating-point precision, and dynamic range.

These laws provide **static, reproducible metrics** for evaluating and fine-tuning LLM reasoning capability.

---

## 1. Introduction

### 1.1 Motivation

Benchmark-based evaluation is costly and prompt-sensitive. Can reasoning capacity be inferred **directly from model weights**?

We focus on attention projections (Q/K/V) and their **singular-value spectra**.

### 1.2 Contributions

- Formalization of **Wang's Three Laws**.  
- Reproducible scripts for verification across LLM families.  
- Correlation between deep-layer SSR and reasoning benchmarks.  
- Practical guidance for training, precision selection, and checkpoint evaluation.

---

## 2. Background

### 2.1 Transformer Attention

\[
\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
\]

with projections $W_Q, W_K, W_V$.

### 2.2 Spectral Analysis

SVD decomposition:

\[
M = U \Sigma V^\top, \quad \Sigma = \text{diag}(\sigma_1,\dots,\sigma_r)
\]

Metrics are computed on $Q$ and $K$ matrices.

---

## 3. Wang's Three Laws

### 3.1 First Law — Spectral Linear Alignment

**Statement:**  
Query and Key spectra are **linearly correlated**:

\[
r(s_q, s_k) \to r_\text{Wang} = 1
\]

**Definition (Wang's Constant):**  
- **Theoretical extreme:** $r_\text{Wang} = 1$  
- **Observed in practice:** $0.94 \sim 0.99$  

**Interpretation:**  
- High Pearson correlation indicates **stable spectral alignment**.  
- Deviations may signal **training anomalies or reduced deep-layer reasoning fidelity**.

**Empirical Evidence:**

| Model       | Median Pearson r (\(r_\text{median}\)) | Layers |
| ----------- | -------------------------------------- | ------ |
| Qwen2.5-14B | 0.974                                  | 48     |
| DeepSeek-R1 | 0.972                                  | 48     |
| LLaMA-3-8B  | 0.967                                  | 32     |
| Gemma-4-E2B | 0.934                                  | 44     |

---

### 3.2 Second Law — Spectral Shape Fidelity

**Statement:**  
Normalized spectral mismatch **SSR** decreases in deep layers:

\[
\text{SSR} = \frac{1}{d_h} \sum_i |\tilde s_{q,i} - \tilde s_{k,i}|, \quad \tilde s = s / \|s\|_2
\]

- **Theoretical extreme:** $\text{SSR}_\text{Wang} = 0$  
- **Observed in practice:** 0.006–0.007

**Empirical Evidence (Qwen2.5 vs DeepSeek-R1):**

| Layer Group | Qwen2.5 SSR | DeepSeek-R1 SSR | Improvement |
| ----------- | ----------- | --------------- | ----------- |
| 0-11        | 0.00708     | 0.00704         | +0.56%      |
| 12-23       | 0.00653     | 0.00645         | +1.17%      |
| 24-35       | 0.00697     | 0.00683         | +1.95%      |
| 36-47       | 0.00676     | 0.00645         | +3.98%      |

> SSR is a **quantitative proxy for reasoning fidelity**.

---

### 3.3 Third Law — Precision-Depth-Logic Criterion

**Statement:**  
Maximum trainable depth $L_\text{max}$ is constrained by:

\[
L_\text{max} = \min\Big(
L_\text{info}, \; L_\text{quant}, \; L_\text{dyn}
\Big)
\]

Where:

- $L_\text{info} = 1/\overline{SSR}$ — information decay limit  
- $L_\text{quant} = 3 \cdot 2^{2m}$ — quantization noise limit ($m$ = mantissa bits)  
- $L_\text{dyn} = \log_\kappa(\text{MaxFinite})$ — dynamic range limit

**Example Table:**

| Format | Mantissa bits $m$ | MaxFinite | $L_\text{dyn}$ |
| ------ | ----------------- | --------- | -------------- |
| FP16   | 10                | 6.55e4    | 16             |
| BF16   | 7                 | 3.39e38   | 128            |

> Explains why ultra-deep models (>40–80 layers) adopt **BF16 or mixed precision**.

---

## 4. Methodology

- **Model Families:** Qwen, DeepSeek, LLaMA, Gemma  
- **Metrics:** Pearson(Q,K), SSR, Deep-layer SSR  
- **Procedure:** Extract weights → compute spectra → calculate metrics → correlate with reasoning benchmarks  

Scripts: `check-gemma.py`, `check-qwen.py`, `check-llama.py`, `check_r1_full.py`, `check_*_v2.py`

---

## 5. Practical Applications

- Benchmark-free reasoning assessment  
- Checkpoint selection based on SSR  
- RL monitoring and spectral fine-tuning  
- Precision planning for ultra-deep training  
- Dynamic range & depth feasibility check

---

## 6. Reproducibility Guide

1. Download model checkpoints:  
   - [LLaMA 3 8B](https://hf-mirror.com/unsloth/llama-3-8b/tree/main)  
   - [Gemma 4 E2B](https://hf-mirror.com/google/gemma-4-E2B/tree/main)  
   - [Qwen2.5-3B](https://hf-mirror.com/Qwen/Qwen2.5-3B/tree/main)  
   - [DeepSeek-R1-Distill-Qwen-14B](https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main)  
   - [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main)  
2. Place models in the same folder as the scripts.  
3. Run scripts:  
```bash
python check-gemma.py
python check-qwen.py
python check-llama.py
python check_r1_full.py
python check_*_v2.py
```

---

## 7. References

1. Vaswani, A., et al. "Attention Is All You Need." NeurIPS, 2017.
2. Shannon, C.E. "A Mathematical Theory of Communication." Bell System Technical Journal, 1948.
3. DeepSeek-AI. DeepSeek-R1 Technical Report, 2025.
4. Meta AI. LLaMA-3 Technical Report, 2024.
5. Qwen Team. Qwen2.5-14B-Instruct Technical Report, 2025.
6. Gemma Team. Gemma-4-E2B Technical Report, 2025.
7. Wang, F. "Wang's Three Laws: A Spectral Theory of Attention Mechanisms in Large Language Models." Zenodo, 2026. DOI: 10.5281/zenodo.19707844


