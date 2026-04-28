import torch
from safetensors.torch import load_file
import numpy as np
from scipy.stats import spearmanr
import glob
import re
import gc

# ---------------------------- 配置区 ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 模型超参数 (DeepSeek‑R1‑Distill‑Qwen‑14B)
D_HEAD = 128
N_Q_HEADS = 40
N_KV_HEADS = 8
GROUP_SIZE = N_Q_HEADS // N_KV_HEADS   # 5

# 所有层 (Qwen 14B 共 48 层)
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

def compute_singular_value_ratio(s_a, s_b):
    """最小二乘拟合 s_a ≈ α s_b，返回 α 和残差 MSE"""
    min_len = min(s_a.shape[0], s_b.shape[0])
    s_a = s_a[:min_len]
    s_b = s_b[:min_len]
    numerator = torch.dot(s_a, s_b)
    denominator = torch.dot(s_b, s_b)
    if denominator == 0:
        return 1.0, 0.0
    alpha = numerator / denominator
    residual = torch.mean((s_a - alpha * s_b) ** 2).item()
    return alpha.item(), residual

def compute_left_vector_alignment(U_a, U_b):
    """列归一化后对应列余弦绝对值的平均"""
    U_a_n = U_a / torch.norm(U_a, dim=0, keepdim=True)
    U_b_n = U_b / torch.norm(U_b, dim=0, keepdim=True)
    cos_mat = torch.abs(torch.mm(U_a_n.T, U_b_n))
    mean_cos = torch.diag(cos_mat).mean().item()
    return mean_cos

def compute_covariance_alignment(W_a, W_b, alpha):
    """比较 W_a W_a^T 与 α² W_b W_b^T 的 Frobenius 相对误差"""
    cov_a = torch.mm(W_a, W_a.T)
    cov_b = torch.mm(W_b, W_b.T)
    diff = cov_a - (alpha ** 2) * cov_b
    rel_error = torch.norm(diff, 'fro') / (torch.norm(cov_a, 'fro') + 1e-8)
    return rel_error.item()

def analyze_layer_from_tensors(wq, wk, wv, layer_idx):
    """分析 Q/K/V 三个投影矩阵的全部两两指标"""
    wq_gpu = wq.to(torch.float32).to(DEVICE)
    wk_gpu = wk.to(torch.float32).to(DEVICE)
    wv_gpu = wv.to(torch.float32).to(DEVICE)

    print(f"\n========== Layer {layer_idx} ==========")
    print(f"  wq shape: {wq.shape}, wk shape: {wk.shape}, wv shape: {wv.shape}")

    for kv_h in range(N_KV_HEADS):
        # ---- 提取 K 和 V 头矩阵 ----
        k_tensor = wk_gpu[kv_h * D_HEAD : (kv_h + 1) * D_HEAD, :]
        v_tensor = wv_gpu[kv_h * D_HEAD : (kv_h + 1) * D_HEAD, :]

        U_k, s_k, _ = torch.linalg.svd(k_tensor, full_matrices=False)
        U_v, s_v, _ = torch.linalg.svd(v_tensor, full_matrices=False)

        # ======== K‑V 指标 ========
        min_len_kv = min(s_k.shape[0], s_v.shape[0])
        s_k_trunc_kv = s_k[:min_len_kv]
        s_v_trunc_kv = s_v[:min_len_kv]

        pearson_kv = compute_pearson_corr_torch(s_k_trunc_kv, s_v_trunc_kv)
        s_k_n = s_k_trunc_kv / torch.norm(s_k_trunc_kv)
        s_v_n = s_v_trunc_kv / torch.norm(s_v_trunc_kv)
        ssr_kv = torch.mean(torch.abs(s_k_n - s_v_n)).item()

        alpha_kv, alpha_res_kv = compute_singular_value_ratio(s_k, s_v)
        cos_ukv = compute_left_vector_alignment(U_k, U_v)

        # V 的谱特征
        max_s_v = s_v.max().item()
        min_s_v = s_v.min().item()
        cond_v = max_s_v / min_s_v if min_s_v > 0 else float('inf')

        cov_err_kv = compute_covariance_alignment(k_tensor, v_tensor, alpha_kv)

        print(f"  KV头 {kv_h} K-V: Pearson(kv)={pearson_kv:+.4f} "
              f"SSR(kv)={ssr_kv:.6f} α(kv)={alpha_kv:.4f}(残差={alpha_res_kv:.2e}) "
              f"cos(Uk,Uv)={cos_ukv:.4f} 协方差误差(kv)={cov_err_kv:.4f} "
              f"σ_max(V)={max_s_v:.3f} cond(V)={cond_v:.1f}")

        # ======== Q 头循环 ========
        start_q = kv_h * GROUP_SIZE
        for q_offset in range(GROUP_SIZE):
            h_idx = start_q + q_offset
            q_tensor = wq_gpu[h_idx * D_HEAD : (h_idx + 1) * D_HEAD, :]
            U_q, s_q, _ = torch.linalg.svd(q_tensor, full_matrices=False)

            # ---- Q‑K 指标 ----
            min_len_qk = min(s_q.shape[0], s_k.shape[0])
            s_q_trunc_qk = s_q[:min_len_qk]
            s_k_trunc_qk = s_k[:min_len_qk]

            pearson_qk = compute_pearson_corr_torch(s_q_trunc_qk, s_k_trunc_qk)
            s_q_n_qk = s_q_trunc_qk / torch.norm(s_q_trunc_qk)
            s_k_n_qk = s_k_trunc_qk / torch.norm(s_k_trunc_qk)
            ssr_qk = torch.mean(torch.abs(s_q_n_qk - s_k_n_qk)).item()

            # Spearman (Q,K) 仅用于校验
            s_q_np = s_q_trunc_qk.cpu().numpy()
            s_k_np = s_k_trunc_qk.cpu().numpy()
            spearman_qk = spearmanr(s_q_np, s_k_np)[0]

            alpha_qk, alpha_res_qk = compute_singular_value_ratio(s_q, s_k)
            cos_u_qk = compute_left_vector_alignment(U_q, U_k)
            cov_err_qk = compute_covariance_alignment(q_tensor, k_tensor, alpha_qk)

            # Q & K 的谱半径与条件数
            max_s_q = s_q.max().item()
            min_s_q = s_q.min().item()
            cond_q = max_s_q / min_s_q if min_s_q > 0 else float('inf')
            max_s_k = s_k.max().item()
            min_s_k = s_k.min().item()
            cond_k = max_s_k / min_s_k if min_s_k > 0 else float('inf')

            # ---- Q‑V 指标（新增） ----
            min_len_qv = min(s_q.shape[0], s_v.shape[0])
            s_q_trunc_qv = s_q[:min_len_qv]
            s_v_trunc_qv = s_v[:min_len_qv]

            pearson_qv = compute_pearson_corr_torch(s_q_trunc_qv, s_v_trunc_qv)
            s_q_n_qv = s_q_trunc_qv / torch.norm(s_q_trunc_qv)
            s_v_n_qv = s_v_trunc_qv / torch.norm(s_v_trunc_qv)
            ssr_qv = torch.mean(torch.abs(s_q_n_qv - s_v_n_qv)).item()

            alpha_qv, alpha_res_qv = compute_singular_value_ratio(s_q, s_v)
            cos_u_qv = compute_left_vector_alignment(U_q, U_v)
            cov_err_qv = compute_covariance_alignment(q_tensor, v_tensor, alpha_qv)

            # 输出所有指标
            print(f"  KV头 {kv_h} | Q头 {h_idx:2d}: "
                  f"Pearson(qk)={pearson_qk:+.4f} Spearman={spearman_qk:+.4f} "
                  f"α(qk)={alpha_qk:.4f}(残差={alpha_res_qk:.2e}) "
                  f"cos(Uq,Uk)={cos_u_qk:.4f} 协方差误差(qk)={cov_err_qk:.4f} "
                  f"SSR(qk)={ssr_qk:.6f} "
                  f"σ_max(Q)={max_s_q:.3f} σ_max(K)={max_s_k:.3f} "
                  f"cond(Q)={cond_q:.1f} cond(K)={cond_k:.1f} "
                  f"Pearson(qv)={pearson_qv:+.4f} SSR(qv)={ssr_qv:.6f} "
                  f"α(qv)={alpha_qv:.4f}(残差={alpha_res_qv:.2e}) "
                  f"cos(Uq,Uv)={cos_u_qv:.4f} 协方差误差(qv)={cov_err_qv:.4f}")

    del wq_gpu, wk_gpu, wv_gpu
    torch.cuda.empty_cache()

# ---------------------------- 主流程 ----------------------------
print(f"\n开始分析 DeepSeek‑R1‑Distill‑Qwen‑14B，共 {len(LAYERS_TO_ANALYZE)} 层（Q/K/V 全面分析）")
analyzed_layers = set()

for file_idx, fpath in enumerate(file_list, 1):
    print(f"\n[分片 {file_idx}/{len(file_list)}] 加载 {fpath} ...")
    weights = load_file(fpath)
    keys = list(weights.keys())

    layers_in_file = set()
    for k in keys:
        m = re.search(r"model\.layers\.(\d+)\.", k)
        if m:
            layers_in_file.add(int(m.group(1)))
    print(f"  该分片包含的层: {sorted(layers_in_file)}")

    to_analyze = [L for L in layers_in_file if L in LAYERS_TO_ANALYZE and L not in analyzed_layers]
    to_analyze.sort()
    if not to_analyze:
        print(f"  没有需要分析的新层，跳过。")
        del weights
        gc.collect()
        continue

    print(f"  将分析层: {to_analyze}")

    for L in to_analyze:
        q_key = f"model.layers.{L}.self_attn.q_proj.weight"
        k_key = f"model.layers.{L}.self_attn.k_proj.weight"
        v_key = f"model.layers.{L}.self_attn.v_proj.weight"
        if q_key not in weights or k_key not in weights or v_key not in weights:
            print(f"    层 {L} 缺少 Q/K/V 键，跳过")
            continue
        wq = weights[q_key]
        wk = weights[k_key]
        wv = weights[v_key]
        analyze_layer_from_tensors(wq, wk, wv, L)
        analyzed_layers.add(L)

    del weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

missing = set(LAYERS_TO_ANALYZE) - analyzed_layers
if missing:
    print(f"\n警告：以下层未找到或未分析：{sorted(missing)}")
else:
    print("\n✅ 所有目标层分析完成（含 Q‑K、K‑V、Q‑V 全指标）。")
