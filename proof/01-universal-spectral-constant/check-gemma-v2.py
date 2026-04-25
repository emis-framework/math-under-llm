import torch
from safetensors.torch import load_file
import numpy as np
from scipy.stats import spearmanr
import re
import os

# ---------------------------- 配置区 ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 权重文件路径（请修改为你的实际路径）
FILE_PATH = "gemma-4-E2B-model.safetensors"
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"请确认文件路径: {FILE_PATH}")

print(f"正在加载模型权重: {FILE_PATH} ...")
weights = load_file(FILE_PATH)

# 需要跳过的辅助张量关键词
SKIP_KEYWORDS = ['output_min', 'output_max', 'input_min', 'input_max', 'scale', 'zero_point']

# ---------------------------- 工具函数 ----------------------------
def compute_pearson_corr_torch(x, y):
    """
    PyTorch GPU 计算 Pearson 相关系数
    """
    xm = x - x.mean()
    ym = y - y.mean()
    r_num = torch.dot(xm, ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    if r_den == 0:
        return 0.0
    return (r_num / r_den).item()

def infer_head_dim(wq_shape0, wk_shape0, candidates=[256, 128, 64, 32]):
    """
    根据 Q 和 K 权重矩阵的行数，推断 head_dim。
    返回推断出的 head_dim，如果都不匹配则返回 1（安全回退）。
    """
    for d in candidates:
        if wq_shape0 % d == 0 and wk_shape0 % d == 0:
            return d
    print(f"警告: 无法推断 head_dim，使用 1。Q_rows={wq_shape0}, K_rows={wk_shape0}")
    return 1

def analyze_layer(layer_idx):
    """
    分析指定层：自动定位 Q/K 权重，推断 head_dim，打印所有头对的 Pearson 和 Spearman。
    """
    # 动态查找当前层的 Q 和 K 权重键
    q_key = None
    k_key = None
    for key in weights.keys():
        if f"layers.{layer_idx}." not in key:
            continue
        if any(skip in key for skip in SKIP_KEYWORDS):
            continue
        tensor = weights[key]
        if tensor.dim() < 2 or tensor.numel() == 0:
            continue
        if "q_proj" in key or ".wq" in key:
            q_key = key
        elif "k_proj" in key or ".wk" in key:
            k_key = key
        # 如果已经找到两个，可以提前退出循环（但为了简单，遍历完也行）
    
    if q_key is None or k_key is None:
        print(f"  跳过层 {layer_idx}: 未找到 Q 或 K 权重")
        return

    # 加载权重并移至 GPU
    wq = weights[q_key].to(torch.float32).to(DEVICE)
    wk = weights[k_key].to(torch.float32).to(DEVICE)
    
    print(f"\n========== Layer {layer_idx} ==========")
    print(f"  wq shape: {wq.shape}, wk shape: {wk.shape}")
    print(f"  键名: Q={q_key}\n        K={k_key}")

    # 推断 head_dim
    head_dim = infer_head_dim(wq.shape[0], wk.shape[0])
    print(f"  推断 head_dim = {head_dim}")
    
    n_q_heads = wq.shape[0] // head_dim
    n_kv_heads = wk.shape[0] // head_dim
    group_size = n_q_heads // n_kv_heads if n_kv_heads > 0 else 1
    print(f"  Q头数={n_q_heads}, KV头数={n_kv_heads}, 组大小={group_size}")
    
    # 逐头计算
    for kv_h in range(n_kv_heads):
        k_tensor = wk[kv_h * head_dim : (kv_h + 1) * head_dim, :]
        _, s_k, _ = torch.linalg.svd(k_tensor, full_matrices=False)
        for q_offset in range(group_size):
            h_idx = kv_h * group_size + q_offset
            if h_idx >= n_q_heads:
                break
            q_tensor = wq[h_idx * head_dim : (h_idx + 1) * head_dim, :]
            _, s_q, _ = torch.linalg.svd(q_tensor, full_matrices=False)
            
            min_len = min(s_q.shape[0], s_k.shape[0])
            s_q_trunc = s_q[:min_len]
            s_k_trunc = s_k[:min_len]
            
            pearson_r = compute_pearson_corr_torch(s_q_trunc, s_k_trunc)
            # Spearman 需要移至 CPU
            s_q_np = s_q_trunc.cpu().numpy()
            s_k_np = s_k_trunc.cpu().numpy()
            spearman_r, _ = spearmanr(s_q_np, s_k_np)
            
            print(f"  KV头 {kv_h} (组内Q偏移 {q_offset}) | Q头索引 {h_idx:2d}: "
                  f"Pearson = {pearson_r:+.4f}, Spearman = {spearman_r:+.4f}")

# ---------------------------- 主流程：获取所有层号 ----------------------------
all_keys = list(weights.keys())
layer_nums = set()
for k in all_keys:
    match = re.search(r'layers\.(\d+)\.', k)
    if match:
        layer_nums.add(int(match.group(1)))
if not layer_nums:
    print("未探测到任何层结构，退出。")
    exit()

layers = sorted(layer_nums)
print(f"探测到总层数: {min(layers)} - {max(layers)}，共 {len(layers)} 层")
print(f"层列表: {layers}")

# 遍历所有层进行分析
for layer in layers:
    try:
        analyze_layer(layer)
    except Exception as e:
        print(f"层 {layer} 处理出错: {e}")
