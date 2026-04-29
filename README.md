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
# Wang's Three Laws: Spectral Signals Inside LLM Attention

A reproducible study showing that reasoning-related signals may be measurable directly from LLM attention weights.

## Core Findings

### Claim 1 — Universal QK Spectral Correlation

Across multiple open-source LLM families, the singular-value spectra of Query (Q) and Key (K) projections show strong alignment.

Observed median Pearson correlation:

> median r > 0.94

---

### Claim 2 — Deep SSR Predicts Reasoning

We define:

SSR (Spectral Shape Residual)

which measures normalized spectral mismatch between Q and K.

We observe:

> lower deep-layer SSR is associated with stronger reasoning benchmarks.

---

### Claim 3 — RL Improves Spectral Structure

RL-tuned reasoning models consistently reduce deep-layer SSR compared with their base variants.

---

## Why This Matters

Current LLM evaluation depends on:

- expensive benchmarks
- contaminated datasets
- long inference runs

Our results suggest a static alternative:

> inspect weights only.

---

## Repository Structure

```text
proof/
  01-universal-spectral-constant/
  02-ssr-why-RL-makes-models-smart/
README.md
README.cn.md
WHITEPAPER.md
WHITEPAPER.cn.md
```

---

## Quick Start

```bash
pip install torch numpy scipy matplotlib
download model
python check_*.py 
python check_*_v2.py
```

---

## Metrics

### Pearson(Q,K Spectrum)

Measures linear alignment between Q and K singular values.

### SSR

Measures normalized shape mismatch.

---

## Reproduce Main Results

---

## Download Models and Reproduce Locally

Place the downloaded model folders in the **same directory as the Python scripts**, then run the verification scripts directly.

Repository layout example:

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

## Model Download Links

### LLaMA 3 8B

[https://hf-mirror.com/unsloth/llama-3-8b/tree/main](https://hf-mirror.com/unsloth/llama-3-8b/tree/main)

### Gemma 4 E2B

[https://hf-mirror.com/google/gemma-4-E2B/tree/main](https://hf-mirror.com/google/gemma-4-E2B/tree/main)

### Qwen2.5 3B

[https://hf-mirror.com/Qwen/Qwen2.5-3B/tree/main](https://hf-mirror.com/Qwen/Qwen2.5-3B/tree/main)

### DeepSeek-R1-Distill-Qwen-14B

[https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main](https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/tree/main)

### Qwen2.5-14B-Instruct

[https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/tree/main)

---

## Direct Run (No Path Arguments Needed)

After placing models in the same folder, simply run:

```bash 
python check-gemma.py
python check-qwen.py
python check-llama.py
python check_*_v2.py
python check_r1_full.py
python qwen-vs-deepseek-all-layers.py
```

---

## What These Scripts Verify

* Q/K singular-value Pearson correlation
* Layer-wise SSR (Spectral Shape Residual)
* Deep-layer spectral alignment trends
* Base vs RL-tuned model comparisons
* Cross-model universality checks

---

## Philosophy

No screenshots.
No benchmark cherry-picking.
No trust required.

Just download the weights, run the scripts, and inspect the matrices yourself.

---


## **Before you close this README —— Bonus：**

1. **"Get to work, lads." — The Monkey King.**  
   GitHub Issue #1: [Verify r = 1](https://github.com/emis-framework/math-under-llm/issues/1)  
   GitHub Issue #2: [Verify SSR = 0](https://github.com/emis-framework/math-under-llm/issues/2)  
   Pick a model. Run the script. Replicate the laws.

2. **If you've made it this far and verified the numbers——consider buying put options on NVIDIA.**  
   If Reasoning = Spectral Fidelity, the demand for brute-force training compute may not be what the market thinks it is.

3. **It's April 29, 2026.**  
   If you're building LLMs in China, sorry，your May Day holiday just evaporated.  
   The sooner you verify, the sooner you submit to arXiv, and the better your chance of being on the list when this thing wins something.

---

> "That's one small step for human intelligence, one giant leap for Artificial Intelligence."
> — Lao Wang, EMIS-FRAMEWORK, Apr 20, 2026

---

## Citation

```bibtex
@misc{wang2026spectral,
  title={Wang's Three Laws: Spectral Signals Inside LLM Attention},
  author={Wang, Fei-Yun},
  year={2026}
}
```
