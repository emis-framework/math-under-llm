import torch
from safetensors.torch import load_file
import numpy as np
from scipy.stats import spearmanr

# 1. 加载文件
FILE_PATH = "model-00001-of-00004.safetensors"
weights = load_file(FILE_PATH)

# --- 自动识别 Key 名 ---
all_keys = list(weights.keys())
# 尝试寻找第 4 层的 Q 权重名
sample_key = [k for k in all_keys if "layers.4." in k and "wq" in k]
if not sample_key:
    # 兼容有些版本叫 layers.4.self_attn.q_proj
    sample_key = [k for k in all_keys if "layers.4." in k and "q_proj" in k]

if not sample_key:
    print("找不到对应键名，请检查 print(all_keys[:20]) 的输出")
    exit()

# 获取通用的前缀和后缀模板
# 假设 sample_key 是 "model.layers.4.self_attn.q_proj.weight"
template_q = sample_key[0].replace("4", "{L}")
template_k = template_q.replace("q_proj", "k_proj").replace("wq", "wk")

print(f"识别到 Key 模板:\nQ: {template_q}\nK: {template_k}")

# 2. 执行分析
N_Q_HEADS = 32
N_KV_HEADS = 8
D_MODEL = 4096
D_HEAD = 128
GROUP_SIZE = 4
L = 7

wq = weights[template_q.format(L=L)]
wk = weights[template_k.format(L=L)]

print(f"\n--- 物理验证结果 (Layer {L}) ---")
for kv_h in range(N_KV_HEADS):
    kh = wk[kv_h * D_HEAD : (kv_h + 1) * D_HEAD, :].to(torch.float32).numpy() #物理动作：你从庞大的权重矩阵中，精确切出了一个具体的 Key 头。
    _, sk, _ = np.linalg.svd(kh) # 数学本质：通过 SVD（奇异值分解）提取该头的奇异值谱 ($s_k$)。这代表了该“字典索引”在空间中各个维度的能量分布。
    
    for q_idx in range(GROUP_SIZE):
        h_idx = kv_h * GROUP_SIZE + q_idx
        qh = wq[h_idx * D_HEAD : (h_idx + 1) * D_HEAD, :].to(torch.float32).numpy()
        _, sq, _ = np.linalg.svd(qh)
        
        inv_sk = 1.0 / (sk + 1e-6)
        corr, _ = spearmanr(sq, inv_sk)
        print(f"Group {kv_h} | Head {h_idx}: 谱相关系数 ρ = {corr:.4f}")
