import torch
from safetensors.torch import load_file
import numpy as np
from scipy.stats import spearmanr
import glob
import re
import os
import gc

# ---------------------------- 配置区 ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 权重文件路径：可以是单个文件，也可以用通配符匹配多个分片
# 示例： "gemma-4-27b-it.safetensors" 或 "gemma-4-27b-model-*.safetensors"
FILE_PATTERN = "gemma-4-E4B-it-model.safetensors"
# FILE_PATTERN = "gemma-4-E2B-model.safetensors"

# 需要跳过的辅助张量关键词（量化参数等）
SKIP_KEYWORDS = ['output_min', 'output_max', 'input_min', 'input_max', 'scale', 'zero_point']

# 自动寻找文件（支持通配符，自然排序）
file_list = sorted(
    glob.glob(FILE_PATTERN),
    key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)]
)
if not file_list:
    raise FileNotFoundError(f"未找到匹配 '{FILE_PATTERN}' 的文件，请修改路径。")
print(f"找到 {len(file_list)} 个文件: {file_list}")

# ---------------------------- 辅助函数 ----------------------------
def compute_pearson_corr_torch(x, y):
    """PyTorch GPU 计算 Pearson 相关系数"""
    xm = x - x.mean()
    ym = y - y.mean()
    r_num = torch.dot(xm, ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    if r_den == 0:
        return 0.0
    return (r_num / r_den).item()

def compute_left_vector_alignment(U_q, U_k):
    """左奇异向量对齐：返回对角线平均绝对余弦相似度"""
    U_q_n = U_q / torch.norm(U_q, dim=0, keepdim=True)
    U_k_n = U_k / torch.norm(U_k, dim=0, keepdim=True)
    cos_mat = torch.abs(torch.mm(U_q_n.T, U_k_n))
    mean_cos = torch.diag(cos_mat).mean().item()
    return mean_cos

def compute_covariance_alignment(W_q, W_k, alpha):
    """协方差对齐误差：||W_q W_q^T - α^2 W_k W_k^T||_F / ||W_q W_q^T||_F"""
    cov_q = torch.mm(W_q, W_q.T)
    cov_k = torch.mm(W_k, W_k.T)
    diff = cov_q - (alpha ** 2) * cov_k
    rel_error = torch.norm(diff, 'fro') / (torch.norm(cov_q, 'fro') + 1e-8)
    return rel_error.item()

def infer_head_dim(wq_rows, wk_rows, candidates=[256, 128, 64, 32]):
    """根据 Q/K 权重行数推断 head_dim"""
    for d in candidates:
        if wq_rows % d == 0 and wk_rows % d == 0:
            return d
    # 回退：取最大公约数的一种简单方法
    for d in range(min(wq_rows, wk_rows), 0, -1):
        if wq_rows % d == 0 and wk_rows % d == 0:
            return d
    return 1

def extract_layer_num(key):
    """从键名提取层号，如 layers.12.xxx -> 12"""
    m = re.search(r'layers\.(\d+)\.', key)
    return int(m.group(1)) if m else -1

def find_weight_key(keys, layer_idx, proj_type):
    """查找指定层和投影类型 (q/k) 的权重键名"""
    for k in keys:
        if f"layers.{layer_idx}." not in k:
            continue
        if any(skip in k for skip in SKIP_KEYWORDS):
            continue
        tensor = weights.get(k)  # 需要在闭包中有 weights
        if tensor is None or tensor.dim() < 2:
            continue
        if proj_type == "q" and ("q_proj" in k or ".wq" in k):
            return k
        if proj_type == "k" and ("k_proj" in k or ".wk" in k):
            return k
    return None

# ---------------------------- 分析一层 ----------------------------
def analyze_layer(wq, wk, layer_idx):
    """分析给定层，输出所有头对的完整指标"""
    # 确保 float32 并在 GPU 上
    wq = wq.to(torch.float32).to(DEVICE)
    wk = wk.to(torch.float32).to(DEVICE)

    print(f"\n========== Layer {layer_idx} ==========")
    print(f"  wq shape: {wq.shape}, wk shape: {wk.shape}")

    # 推断架构参数
    head_dim = infer_head_dim(wq.shape[0], wk.shape[0])
    n_q_heads = wq.shape[0] // head_dim
    n_kv_heads = wk.shape[0] // head_dim
    group_size = n_q_heads // n_kv_heads
    print(f"  推断 head_dim={head_dim}, Q头数={n_q_heads}, KV头数={n_kv_heads}, 组大小={group_size}")

    # 逐 KV 头，然后组内 Q 头
    for kv_h in range(n_kv_heads):
        k_tensor = wk[kv_h * head_dim : (kv_h + 1) * head_dim, :]
        # 预先计算 K 的 SVD（在同组 Q 头间复用）
        U_k, s_k, _ = torch.linalg.svd(k_tensor, full_matrices=False)

        start_q = kv_h * group_size
        for q_offset in range(group_size):
            h_idx = start_q + q_offset
            if h_idx >= n_q_heads:
                break
            q_tensor = wq[h_idx * head_dim : (h_idx + 1) * head_dim, :]
            U_q, s_q, _ = torch.linalg.svd(q_tensor, full_matrices=False)

            # 截断到相同长度
            min_len = min(s_q.shape[0], s_k.shape[0])
            s_q_trunc = s_q[:min_len]
            s_k_trunc = s_k[:min_len]

            # Pearson
            pearson_r = compute_pearson_corr_torch(s_q_trunc, s_k_trunc)

            # Spearman
            s_q_np = s_q_trunc.cpu().numpy()
            s_k_np = s_k_trunc.cpu().numpy()
            spearman_r, _ = spearmanr(s_q_np, s_k_np)

            # ---------- 新增：最大奇异值 & 条件数 ----------
            max_s_q = s_q.max().item()
            max_s_k = s_k.max().item()
            min_s_q = s_q.min().item()
            min_s_k = s_k.min().item()
            cond_q = max_s_q / min_s_q if min_s_q > 0 else float('inf')
            cond_k = max_s_k / min_s_k if min_s_k > 0 else float('inf')
            # -----------------------------------------------

            # SSR (归一化后 L1 距离)
            s_q_norm = s_q_trunc / torch.norm(s_q_trunc)
            s_k_norm = s_k_trunc / torch.norm(s_k_trunc)
            ssr = torch.mean(torch.abs(s_q_norm - s_k_norm)).item()

            # α 及其残差 (s_q ≈ α * s_k)
            numerator = torch.dot(s_q_trunc, s_k_trunc)
            denominator = torch.dot(s_k_trunc, s_k_trunc)
            if denominator == 0:
                alpha = 1.0
                alpha_res = 0.0
            else:
                alpha = (numerator / denominator).item()
                alpha_res = torch.mean((s_q_trunc - alpha * s_k_trunc) ** 2).item()

            # 左奇异向量对齐
            cos_u = compute_left_vector_alignment(U_q, U_k)

            # 协方差对齐误差
            cov_err = compute_covariance_alignment(q_tensor, k_tensor, alpha)

            print(f"  KV头 {kv_h} | Q头 {h_idx:2d}: "
                  f"Pearson={pearson_r:+.4f} Spearman={spearman_r:+.4f} "
                  f"α={alpha:.4f}(残差={alpha_res:.2e}) "
                  f"cos(Uq,Uk)={cos_u:.4f} 协方差误差={cov_err:.4f} SSR={ssr:.6f} "
                  f"σ_max(Q)={max_s_q:.3f} σ_max(K)={max_s_k:.3f} "
                  f"cond(Q)={cond_q:.1f} cond(K)={cond_k:.1f}")

# ---------------------------- 主流程：逐文件加载，跨层分析 ----------------------------
print("\n开始分析...")
analyzed_layers = set()

for fpath in file_list:
    print(f"\n[加载] {fpath}")
    weights = load_file(fpath)
    keys = list(weights.keys())

    # 提取该文件包含的层号
    file_layers = set()
    for k in keys:
        if any(skip in k for skip in SKIP_KEYWORDS):
            continue
        l = extract_layer_num(k)
        if l >= 0:
            file_layers.add(l)
    if not file_layers:
        print("  未发现任何层，跳过")
        del weights
        gc.collect()
        continue

    print(f"  覆盖层: {sorted(file_layers)}")
    # 仅分析尚未处理过的层
    for layer_idx in sorted(file_layers):
        if layer_idx in analyzed_layers:
            continue
        # 查找 Q/K 键
        q_key = find_weight_key(keys, layer_idx, "q")
        k_key = find_weight_key(keys, layer_idx, "k")
        if q_key is None or k_key is None:
            print(f"  层 {layer_idx} 缺少 Q 或 K 权重，跳过")
            continue
        wq = weights[q_key]
        wk = weights[k_key]
        try:
            analyze_layer(wq, wk, layer_idx)
            analyzed_layers.add(layer_idx)
        except Exception as e:
            print(f"  层 {layer_idx} 分析出错: {e}")

    del weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

missing = set(range(min(analyzed_layers, default=0), max(analyzed_layers, default=0)+1)) - analyzed_layers
if missing:
    print(f"\n⚠️ 未分析的层: {sorted(missing)}")
else:
    print("\n✅ 所有检测到的层已完成分析。")
