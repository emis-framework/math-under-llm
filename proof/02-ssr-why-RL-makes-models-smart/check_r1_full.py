import torch
from safetensors.torch import load_file
import numpy as np
from scipy.stats import spearmanr
import glob
import re
import gc
import os

# ---------------------------- 配置区 ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 模型超参数 (DeepSeek-R1-Distill-Qwen-14B)
D_HEAD = 128
N_Q_HEADS = 40
N_KV_HEADS = 8
GROUP_SIZE = N_Q_HEADS // N_KV_HEADS   # 5

# 所有层 (Qwen 14B 通常 48 层)
LAYERS_TO_ANALYZE = range(48)

# 自动查找分片
file_pattern = "*DeepSeek-R1-Distill-Qwen-14B-model*.safetensors"
file_list = sorted(glob.glob(file_pattern))
if not file_list:
    raise FileNotFoundError(f"未找到匹配 '{file_pattern}' 的分片文件，请检查路径。")
print(f"找到 {len(file_list)} 个分片: {file_list}")

# ---------------------------- 工具函数 ----------------------------
def compute_pearson_corr_torch(x, y):
    """GPU 计算 Pearson 相关系数"""
    xm = x - x.mean()
    ym = y - y.mean()
    r_num = torch.dot(xm, ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    if r_den == 0:
        return 0.0
    return (r_num / r_den).item()

def compute_singular_value_ratio(s_q, s_k):
    """最小二乘拟合 s_q ≈ α s_k，返回 α 和残差均值"""
    min_len = min(s_q.shape[0], s_k.shape[0])
    s_q = s_q[:min_len]
    s_k = s_k[:min_len]
    numerator = torch.dot(s_q, s_k)
    denominator = torch.dot(s_k, s_k)
    if denominator == 0:
        return 1.0, 0.0
    alpha = numerator / denominator
    residual = torch.mean((s_q - alpha * s_k) ** 2).item()
    return alpha.item(), residual

def compute_left_vector_alignment(U_q, U_k):
    """计算左奇异向量矩阵列平均余弦相似度（绝对值）"""
    U_q_n = U_q / torch.norm(U_q, dim=0, keepdim=True)
    U_k_n = U_k / torch.norm(U_k, dim=0, keepdim=True)
    cos_mat = torch.abs(torch.mm(U_q_n.T, U_k_n))
    mean_cos = torch.diag(cos_mat).mean().item()
    return mean_cos

def compute_covariance_alignment(W_q, W_k, alpha):
    """比较 Wq Wq^T 与 α² Wk Wk^T 的相对 Frobenius 误差"""
    cov_q = torch.mm(W_q, W_q.T)
    cov_k = torch.mm(W_k, W_k.T)
    diff = cov_q - (alpha ** 2) * cov_k
    rel_error = torch.norm(diff, 'fro') / (torch.norm(cov_q, 'fro') + 1e-8)
    return rel_error.item()

def analyze_layer_from_tensors(wq, wk, layer_idx):
    """分析指定层的完整指标"""
    wq_gpu = wq.to(torch.float32).to(DEVICE)
    wk_gpu = wk.to(torch.float32).to(DEVICE)

    print(f"\n========== Layer {layer_idx} ==========")
    print(f"  wq shape: {wq.shape}, wk shape: {wk.shape}")

    for kv_h in range(N_KV_HEADS):
        # K 头矩阵
        k_tensor = wk_gpu[kv_h * D_HEAD : (kv_h + 1) * D_HEAD, :]
        U_k, s_k, _ = torch.linalg.svd(k_tensor, full_matrices=False)

        start_q = kv_h * GROUP_SIZE
        for q_offset in range(GROUP_SIZE):
            h_idx = start_q + q_offset
            q_tensor = wq_gpu[h_idx * D_HEAD : (h_idx + 1) * D_HEAD, :]
            U_q, s_q, _ = torch.linalg.svd(q_tensor, full_matrices=False)

            # 奇异值相关性
            min_len = min(s_q.shape[0], s_k.shape[0])
            s_q_trunc = s_q[:min_len]
            s_k_trunc = s_k[:min_len]
            s_q_n = s_q_trunc / torch.norm(s_q_trunc)
            s_k_n = s_k_trunc / torch.norm(s_k_trunc)
            ssr = torch.mean(torch.abs(s_q_n - s_k_n)).item()
            pearson = compute_pearson_corr_torch(s_q_trunc, s_k_trunc)
            s_q_np = s_q_trunc.cpu().numpy()
            s_k_np = s_k_trunc.cpu().numpy()
            spearman = spearmanr(s_q_np, s_k_np)[0]

            # ---------- 新增：最大奇异值 & 条件数 ----------
            max_s_q = s_q.max().item()
            max_s_k = s_k.max().item()
            min_s_q = s_q.min().item()
            min_s_k = s_k.min().item()
            cond_q = max_s_q / min_s_q if min_s_q > 0 else float('inf')
            cond_k = max_s_k / min_s_k if min_s_k > 0 else float('inf')
            # -----------------------------------------------

            # 比例因子 α 和残差
            alpha, alpha_res = compute_singular_value_ratio(s_q, s_k)

            # 左奇异向量对齐
            cos_u = compute_left_vector_alignment(U_q, U_k)

            # 协方差对齐误差
            cov_err = compute_covariance_alignment(q_tensor, k_tensor, alpha)

            print(f"  KV头 {kv_h} | Q头 {h_idx:2d}: "
                  f"Pearson={pearson:+.4f} Spearman={spearman:+.4f} "
                  f"α={alpha:.4f}(残差={alpha_res:.2e}) "
                  f"cos(Uq,Uk)={cos_u:.4f} 协方差误差={cov_err:.4f} SSR={ssr:.6f} "
                  f"σ_max(Q)={max_s_q:.3f} σ_max(K)={max_s_k:.3f} "
                  f"cond(Q)={cond_q:.1f} cond(K)={cond_k:.1f}")

    # 释放 GPU 内存
    del wq_gpu, wk_gpu
    torch.cuda.empty_cache()

# ---------------------------- 主流程：逐分片加载 ----------------------------
print(f"\n开始分析 DeepSeek-R1-Distill-Qwen-14B，共 {len(LAYERS_TO_ANALYZE)} 层")
analyzed_layers = set()

for file_idx, fpath in enumerate(file_list, 1):
    print(f"\n[分片 {file_idx}/{len(file_list)}] 加载 {fpath} ...")
    weights = load_file(fpath)
    keys = list(weights.keys())

    # 提取此分片包含的层号
    layers_in_file = set()
    for k in keys:
        m = re.search(r"model\.layers\.(\d+)\.", k)
        if m:
            layers_in_file.add(int(m.group(1)))
    print(f"  该分片包含的层: {sorted(layers_in_file)}")

    # 需要分析的层（在目标范围内且未分析过）
    to_analyze = [L for L in layers_in_file if L in LAYERS_TO_ANALYZE and L not in analyzed_layers]
    to_analyze.sort() 
    if not to_analyze:
        print(f"  没有需要分析的新层，跳过。")
        del weights
        gc.collect()
        continue

    print(f"  将分析层: {to_analyze}")

    # 逐层提取并分析
    for L in to_analyze:
        q_key = f"model.layers.{L}.self_attn.q_proj.weight"
        k_key = f"model.layers.{L}.self_attn.k_proj.weight"
        if q_key not in weights or k_key not in weights:
            print(f"    层 {L} 缺少 Q/K 键，跳过")
            continue
        wq = weights[q_key]
        wk = weights[k_key]
        analyze_layer_from_tensors(wq, wk, L)
        analyzed_layers.add(L)

    # 释放当前分片
    del weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 检查是否所有层都分析了
missing = set(LAYERS_TO_ANALYZE) - analyzed_layers
if missing:
    print(f"\n警告：以下层未找到或未分析：{sorted(missing)}")
else:
    print("\n✅ 所有目标层分析完成。")
