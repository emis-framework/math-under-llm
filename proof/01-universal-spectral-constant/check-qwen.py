import torch
from safetensors.torch import load_file
import numpy as np
from scipy.stats import spearmanr, pearsonr

# 1. 指向 Qwen2.5-3B 的权重文件
FILE_PATH = "qwen-model-00001-of-00002.safetensors" 
print(f"正在读取模型: {FILE_PATH}...")
weights = load_file(FILE_PATH)

# 2. 自动定位 Key (Qwen 的命名规范通常是 model.layers.N.self_attn.q_proj.weight)
all_keys = list(weights.keys())
L = 7  # 采样中间层

# 寻找匹配当前层的 Q 和 K
q_key = f"model.layers.{L}.self_attn.q_proj.weight"
k_key = f"model.layers.{L}.self_attn.k_proj.weight"

# 如果找不到，尝试打印前10个key帮你手动对齐
if q_key not in weights:
    print("自动匹配失败，请检查以下 Key 列表并手动修改代码：")
    print(all_keys[:15])
    exit()

print(f"检测到 Key 对齐: Q -> {q_key} | K -> {k_key}")

# 3. 提取权重并计算
wq_all = weights[q_key].to(torch.float32).numpy()
wk_all = weights[k_key].to(torch.float32).numpy()

# Qwen2.5-3B 参数: d_model=2048, heads=16, d_head=128
d_head = 128
n_q_heads = wq_all.shape[0] // d_head
n_k_heads = wk_all.shape[0] // d_head
group_size = n_q_heads // n_k_heads

print(f"\n--- Qwen2.5 物理对偶验证 (Layer {L}) ---")

for kv_h in range(n_k_heads):
    kh = wk_all[kv_h*d_head : (kv_h+1)*d_head, :]
    _, sk, _ = np.linalg.svd(kh)
    
    for q_idx in range(group_size):
        h_idx = kv_h * group_size + q_idx
        qh = wq_all[h_idx*d_head : (h_idx+1)*d_head, :]
        _, sq, _ = np.linalg.svd(qh)
        
        # 验证谱相关性 (sq vs sk)
        pearson_r, _ = pearsonr(sq, sk)
        spearman_r, _ = spearmanr(sq, sk)
        # 验证矩阵差异
        diff = np.linalg.norm(qh - kh) / (np.linalg.norm(kh) + 1e-9)
        
        if h_idx < 4: # 打印前几个头看规律
            print(f"Head {h_idx}: Pearson = {pearson_r:+.4f}, Spearman = {spearman_r:+.4f} | diff = {diff:.4f}")
