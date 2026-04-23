import torch
from safetensors.torch import load_file
import numpy as np
from scipy.stats import spearmanr
import re

# 1. 指向你的 Gemma-4 权重文件
FILE_PATH = "gemma-4-E2B-model.safetensors"
print(f"正在扫描权重指纹: {FILE_PATH}...")
weights = load_file(FILE_PATH)

# --- 自动探测逻辑 ---
all_keys = list(weights.keys())
print("\n权重 Key 预览 (前 5 个):")
for k in all_keys[:5]: print(f"  {k}")

# 提取层号
layer_nums = []
for k in all_keys:
    match = re.search(r'layers\.(\d+)\.', k)
    if match:
        layer_nums.append(int(match.group(1)))

if not layer_nums:
    print("\n[错误] 未探测到层结构，请确认文件是否完整。")
    exit()

layers = sorted(list(set(layer_nums)))
L = layers[len(layers)//2] # 选中间层，物理特性最稳定
print(f"\n探测到层数范围: {min(layers)} - {max(layers)}, 选定分析层: {L}")

# 定位 Q 和 K
q_key_list = [k for k in all_keys if f"layers.{L}." in k and ("q_proj" in k or ".wq" in k)]
k_key_list = [k for k in all_keys if f"layers.{L}." in k and ("k_proj" in k or ".wk" in k)]

if not q_key_list or not k_key_list:
    print(f"在层 {L} 中找不到 Q 或 K。")
    exit()

q_key = q_key_list[0]
k_key = k_key_list[0]
print(f"锁定目标:\n  Q -> {q_key}\n  K -> {k_key}")

# --- 提取并转换数据 ---
wq = weights[q_key].to(torch.float32).numpy()
wk = weights[k_key].to(torch.float32).numpy()

# 自动适配 Head 维度：优先尝试 256（Gemma 标配）和 128
D_HEAD = 128 # 默认值
for d_test in [256, 128]:
    if wk.shape[0] % d_test == 0:
        D_HEAD = d_test
        break
else:
    D_HEAD = wk.shape[0] 

n_kv_heads = wk.shape[0] // D_HEAD
print(f"识别到架构特征: D_HEAD = {D_HEAD}, KV_Heads = {n_kv_heads}")

print(f"\n--- Google Gemma-4 物理对偶验证 (Layer {L}) ---")
results = []
for h in range(min(n_kv_heads, 8)): # 验证前 8 个头
    kh = wk[h*D_HEAD : (h+1)*D_HEAD, :]
    qh = wq[h*D_HEAD : (h+1)*D_HEAD, :] 
    
    _, sk, _ = np.linalg.svd(kh)
    _, sq, _ = np.linalg.svd(qh)
    
    rho, _ = spearmanr(sq, sk)
    diff = np.linalg.norm(qh - kh) / (np.linalg.norm(kh) + 1e-9)
    results.append(rho)
    print(f"Head {h:02d}: ρ = {rho:.6f} | diff = {diff:.4f}")

print(f"\n--- 最终结论 ---")
print(f"平均谱相关性 ρ: {np.mean(results):.6f}")
