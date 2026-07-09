import torch
from safetensors.torch import load_file
import numpy as np
from scipy.stats import spearmanr
import glob
import re
import gc
import os
import json

# ---------------------------- 配置区 ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 模型路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen3.5-0.8B")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
FILE_PATTERN = os.path.join(MODEL_DIR, "model.safetensors-*.safetensors")

# 加载配置文件
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

text_config = config["text_config"]
LAYER_TYPES = text_config["layer_types"]  # 每层的类型: linear_attention 或 full_attention
NUM_LAYERS = text_config["num_hidden_layers"]
D_HEAD = text_config["head_dim"]
HIDDEN_SIZE = text_config["hidden_size"]
N_Q_HEADS_FULL = text_config["num_attention_heads"]  # full attention 的 Q 头数
N_KV_HEADS_FULL = text_config["num_key_value_heads"]   # full attention 的 KV 头数

# linear attention 的参数
N_KV_HEADS_LINEAR = text_config["linear_num_key_heads"]
N_Q_HEADS_LINEAR = N_KV_HEADS_LINEAR  # linear attention 中 Q 和 KV 头数相同
D_HEAD_LINEAR = text_config["linear_key_head_dim"]

print(f"\n=== Qwen 3.5 模型配置 ===")
print(f"总层数: {NUM_LAYERS}")
print(f"线性注意力层 (Linear Attention):")
print(f"  - head_dim: {D_HEAD_LINEAR}")
print(f"  - num_key_value_heads: {N_KV_HEADS_LINEAR}")
print(f"全注意力层 (Full Attention):")
print(f"  - head_dim: {D_HEAD}")
print(f"  - num_query_heads: {N_Q_HEADS_FULL}")
print(f"  - num_key_value_heads: {N_KV_HEADS_FULL}")
print(f"隐藏层维度: {HIDDEN_SIZE}")

# 所有层
LAYERS_TO_ANALYZE = range(NUM_LAYERS)

# 打印层类型分布
linear_layers = [i for i, t in enumerate(LAYER_TYPES) if t == "linear_attention"]
full_layers = [i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"]
print(f"\n线性注意力层: {linear_layers}")
print(f"全注意力层: {full_layers}")

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

# ==================== 分析 Linear Attention 层 ====================
def analyze_linear_attention_layer(w_in_proj_qkv, layer_idx):
    """
    分析 linear attention 层。
    
    Qwen 3.5 的 linear attention 使用 in_proj_qkv 权重，形状为 [6144, 1024]
    这是因为: (linear_num_key_heads * linear_value_head_dim * 3) = 16*128*3 = 6144
    
    该权重矩阵包含了 Q, K, V 的投影（按顺序拼接）。
    我们将其分解为 Q 和 K 部分，然后对每个头的权重矩阵做 SVD 分析。
    
    对于 linear attention，Q 和 K 的结构是:
    - Q: [num_heads * head_dim, hidden_size] = [2048, 1024] (16 heads * 128 dim)
    - K: [num_heads * head_dim, hidden_size] = [2048, 1024] (16 heads * 128 dim)
    """
    w_qkv = w_in_proj_qkv.to(torch.float32).to(DEVICE)
    # 分解: Q, K, V 各占 1/3
    # 形状: [6144, 1024] -> 分为 [2048, 1024] 三份
    chunk_size = w_qkv.shape[0] // 3
    w_q = w_qkv[:chunk_size, :]        # Q 投影
    w_k = w_qkv[chunk_size:2*chunk_size, :]  # K 投影
    
    print(f"\n========== Linear Attention Layer {layer_idx} ==========")
    print(f"  w_q shape: {w_q.shape}, w_k shape: {w_k.shape}")
    print(f"  (每个头的维度: {D_HEAD_LINEAR}, 头数: {N_Q_HEADS_LINEAR})")
    
    pearson_list = []
    spearman_list = []
    ssr_list = []
    
    for h_idx in range(N_Q_HEADS_LINEAR):
        # 提取单个头的权重矩阵
        q_head = w_q[h_idx * D_HEAD_LINEAR : (h_idx + 1) * D_HEAD_LINEAR, :]
        k_head = w_k[h_idx * D_HEAD_LINEAR : (h_idx + 1) * D_HEAD_LINEAR, :]
        
        # SVD
        U_q, s_q, _ = torch.linalg.svd(q_head, full_matrices=False)
        U_k, s_k, _ = torch.linalg.svd(k_head, full_matrices=False)
        
        # 对齐长度
        min_len = min(s_q.shape[0], s_k.shape[0])
        s_q_trunc = s_q[:min_len]
        s_k_trunc = s_k[:min_len]
        
        # 归一化谱
        s_q_n = s_q_trunc / torch.norm(s_q_trunc)
        s_k_n = s_k_trunc / torch.norm(s_k_trunc)
        
        # 计算指标
        ssr = torch.mean(torch.abs(s_q_n - s_k_n)).item()
        pearson = compute_pearson_corr_torch(s_q_trunc, s_k_trunc)
        s_q_np = s_q_trunc.cpu().numpy()
        s_k_np = s_k_trunc.cpu().numpy()
        spearman = spearmanr(s_q_np, s_k_np)[0]
        
        # 额外指标
        alpha, alpha_res = compute_singular_value_ratio(s_q_trunc, s_k_trunc)
        cos_u = compute_left_vector_alignment(U_q, U_k)
        cov_err = compute_covariance_alignment(q_head, k_head, alpha)
        
        pearson_list.append(pearson)
        spearman_list.append(spearman)
        ssr_list.append(ssr)
        
        print(f"  头 {h_idx:2d}: Pearson={pearson:+.4f} Spearman={spearman:+.4f} "
              f"SSR={ssr:.6f} α={alpha:.4f}(残差={alpha_res:.2e}) "
              f"cos(Uq,Uk)={cos_u:.4f} 协方差误差={cov_err:.4f}")
    
    del w_qkv, w_q, w_k
    
    return pearson_list, spearman_list, ssr_list

# ==================== 分析 Full Attention 层 ====================
def analyze_full_attention_layer(w_q, w_k, layer_idx):
    """
    分析 full attention 层（标准自注意力）。
    
    Q 形状: [num_query_heads * head_dim, hidden_size] = [8*256, 1024] = [2048, 1024]
    K 形状: [num_key_value_heads * head_dim, hidden_size] = [2*256, 1024] = [512, 1024]
    
    GQA (Grouped Query Attention): 8 个 Q 头共享 2 个 KV 头，每组 4 个 Q 头。
    """
    w_q_gpu = w_q.to(torch.float32).to(DEVICE)
    w_k_gpu = w_k.to(torch.float32).to(DEVICE)
    
    print(f"\n========== Full Attention Layer {layer_idx} ==========")
    print(f"  w_q shape: {w_q.shape}, w_k shape: {w_k.shape}")
    print(f"  Q头数={N_Q_HEADS_FULL}, KV头数={N_KV_HEADS_FULL}, 组大小={N_Q_HEADS_FULL // N_KV_HEADS_FULL}")
    
    pearson_list = []
    spearman_list = []
    ssr_list = []
    group_size = N_Q_HEADS_FULL // N_KV_HEADS_FULL
    
    for kv_h in range(N_KV_HEADS_FULL):
        k_tensor = w_k_gpu[kv_h * D_HEAD : (kv_h + 1) * D_HEAD, :]
        U_k, s_k, _ = torch.linalg.svd(k_tensor, full_matrices=False)
        
        for q_offset in range(group_size):
            h_idx = kv_h * group_size + q_offset
            q_tensor = w_q_gpu[h_idx * D_HEAD : (h_idx + 1) * D_HEAD, :]
            U_q, s_q, _ = torch.linalg.svd(q_tensor, full_matrices=False)
            
            # 对齐长度
            min_len = min(s_q.shape[0], s_k.shape[0])
            s_q_trunc = s_q[:min_len]
            s_k_trunc = s_k[:min_len]
            
            # 归一化谱
            s_q_n = s_q_trunc / torch.norm(s_q_trunc)
            s_k_n = s_k_trunc / torch.norm(s_k_trunc)
            
            # 计算指标
            ssr = torch.mean(torch.abs(s_q_n - s_k_n)).item()
            pearson = compute_pearson_corr_torch(s_q_trunc, s_k_trunc)
            s_q_np = s_q_trunc.cpu().numpy()
            s_k_np = s_k_trunc.cpu().numpy()
            spearman = spearmanr(s_q_np, s_k_np)[0]
            
            # 额外指标
            alpha, alpha_res = compute_singular_value_ratio(s_q_trunc, s_k_trunc)
            cos_u = compute_left_vector_alignment(U_q, U_k)
            cov_err = compute_covariance_alignment(q_tensor, k_tensor, alpha)
            
            pearson_list.append(pearson)
            spearman_list.append(spearman)
            ssr_list.append(ssr)
            
            print(f"  KV头 {kv_h} | Q头 {h_idx:2d}: "
                  f"Pearson={pearson:+.4f} Spearman={spearman:+.4f} "
                  f"SSR={ssr:.6f} α={alpha:.4f}(残差={alpha_res:.2e}) "
                  f"cos(Uq,Uk)={cos_u:.4f} 协方差误差={cov_err:.4f}")
    
    del w_q_gpu, w_k_gpu
    
    return pearson_list, spearman_list, ssr_list

# ==================== 主流程 ====================
print(f"\n{'='*60}")
print(f"开始分析 Qwen 3.5 模型")
print(f"{'='*60}")

# 查找所有分片
file_list = sorted(glob.glob(FILE_PATTERN))
if not file_list:
    raise FileNotFoundError(f"未找到匹配 '{FILE_PATTERN}' 的分片文件")
print(f"\n找到 {len(file_list)} 个分片: {file_list}")

# 汇总统计
all_pearson_linear = []
all_spearman_linear = []
all_ssr_linear = []
all_pearson_full = []
all_spearman_full = []
all_ssr_full = []

analyzed_layers = set()

for file_idx, fpath in enumerate(file_list, 1):
    print(f"\n{'='*60}")
    print(f"[分片 {file_idx}/{len(file_list)}] 加载 {fpath}")
    print(f"{'='*60}")
    weights = load_file(fpath)
    keys = list(weights.keys())
    
    # 提取此分片包含的层
    layers_in_file = set()
    for k in keys:
        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            layers_in_file.add(int(m.group(1)))
    print(f"  该分片包含的层: {sorted(layers_in_file)}")
    
    # 需要分析的层
    to_analyze = [L for L in layers_in_file if L in LAYERS_TO_ANALYZE and L not in analyzed_layers]
    to_analyze.sort()
    if not to_analyze:
        print(f"  没有需要分析的新层，跳过。")
        del weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        continue
    
    print(f"  将分析层: {to_analyze}")
    
    # 逐层分析
    for L in to_analyze:
        layer_type = LAYER_TYPES[L]
        print(f"\n>>> 分析层 {L} (类型: {layer_type})")
        
        try:
            if layer_type == "linear_attention":
                # 查找 linear attention 的权重
                qkv_key = f"model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight"
                if qkv_key not in weights:
                    print(f"    警告: 未找到 {qkv_key}，跳过")
                    continue
                
                w_qkv = weights[qkv_key]
                pearson_list, spearman_list, ssr_list = analyze_linear_attention_layer(w_qkv, L)
                
                all_pearson_linear.extend(pearson_list)
                all_spearman_linear.extend(spearman_list)
                all_ssr_linear.extend(ssr_list)
                
            elif layer_type == "full_attention":
                # 查找 full attention 的权重
                q_key = f"model.language_model.layers.{L}.self_attn.q_proj.weight"
                k_key = f"model.language_model.layers.{L}.self_attn.k_proj.weight"
                
                if q_key not in weights or k_key not in weights:
                    print(f"    警告: 未找到 Q/K 权重，跳过")
                    continue
                
                w_q = weights[q_key]
                w_k = weights[k_key]
                pearson_list, spearman_list, ssr_list = analyze_full_attention_layer(w_q, w_k, L)
                
                all_pearson_full.extend(pearson_list)
                all_spearman_full.extend(spearman_list)
                all_ssr_full.extend(ssr_list)
            
            analyzed_layers.add(L)
            
        except Exception as e:
            print(f"    层 {L} 处理出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 释放当前分片
    del weights
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==================== 汇总报告 ====================
print(f"\n\n{'='*60}")
print(f"Qwen 3.5 模型谱分析汇总报告")
print(f"{'='*60}")

if all_pearson_linear:
    print(f"\n--- 线性注意力层 (Linear Attention, {len(linear_layers)} 层) ---")
    print(f"  Pearson 相关系数:  均值={np.mean(all_pearson_linear):.4f}, "
          f"最小={np.min(all_pearson_linear):.4f}, 最大={np.max(all_pearson_linear):.4f}")
    print(f"  Spearman 秩相关:   均值={np.mean(all_spearman_linear):.4f}, "
          f"最小={np.min(all_spearman_linear):.4f}, 最大={np.max(all_spearman_linear):.4f}")
    print(f"  SSR (谱形状差异):  均值={np.mean(all_ssr_linear):.6f}, "
          f"最小={np.min(all_ssr_linear):.6f}, 最大={np.max(all_ssr_linear):.6f}")
    
    # Wang's Law 评估
    pearson_linear_mean = np.mean(all_pearson_linear)
    ssr_linear_mean = np.mean(all_ssr_linear)
    print(f"\n  Wang's 第一定律 (r→1):  {'✓ 通过' if pearson_linear_mean > 0.95 else '⚠ 需关注'} "
          f"(r={pearson_linear_mean:.4f})")
    print(f"  Wang's 第二定律 (SSR→0): {'✓ 通过' if ssr_linear_mean < 0.01 else '⚠ 需关注'} "
          f"(SSR={ssr_linear_mean:.6f})")

if all_pearson_full:
    print(f"\n--- 全注意力层 (Full Attention, {len(full_layers)} 层) ---")
    print(f"  Pearson 相关系数:  均值={np.mean(all_pearson_full):.4f}, "
          f"最小={np.min(all_pearson_full):.4f}, 最大={np.max(all_pearson_full):.4f}")
    print(f"  Spearman 秩相关:   均值={np.mean(all_spearman_full):.4f}, "
          f"最小={np.min(all_spearman_full):.4f}, 最大={np.max(all_spearman_full):.4f}")
    print(f"  SSR (谱形状差异):  均值={np.mean(all_ssr_full):.6f}, "
          f"最小={np.min(all_ssr_full):.6f}, 最大={np.max(all_ssr_full):.6f}")
    
    pearson_full_mean = np.mean(all_pearson_full)
    ssr_full_mean = np.mean(all_ssr_full)
    print(f"\n  Wang's 第一定律 (r→1):  {'✓ 通过' if pearson_full_mean > 0.95 else '⚠ 需关注'} "
          f"(r={pearson_full_mean:.4f})")
    print(f"  Wang's 第二定律 (SSR→0): {'✓ 通过' if ssr_full_mean < 0.01 else '⚠ 需关注'} "
          f"(SSR={ssr_full_mean:.6f})")

# 总体评估
all_pearson = all_pearson_linear + all_pearson_full
all_spearman = all_spearman_linear + all_spearman_full
all_ssr = all_ssr_linear + all_ssr_full

if all_pearson:
    print(f"\n--- 总体评估 (所有头) ---")
    print(f"  总头数: {len(all_pearson)}")
    print(f"  Pearson 相关系数:  均值={np.mean(all_pearson):.4f}, "
          f"最小={np.min(all_pearson):.4f}, 最大={np.max(all_pearson):.4f}")
    print(f"  Spearman 秩相关:   均值={np.mean(all_spearman):.4f}, "
          f"最小={np.min(all_spearman):.4f}, 最大={np.max(all_spearman):.4f}")
    print(f"  SSR (谱形状差异):  均值={np.mean(all_ssr):.6f}, "
          f"最小={np.min(all_ssr):.6f}, 最大={np.max(all_ssr):.6f}")
    
    print(f"\n  {'='*60}")
    if np.mean(all_pearson) > 0.95 and np.mean(all_ssr) < 0.01:
        print(f"  ✅ 结论: Qwen 3.5 模型符合 Wang's Laws!")
        print(f"     理论成立: 模型具有良好的推理能力特征")
    else:
        print(f"  ⚠ 结论: 部分指标偏离 Wang's Laws")
        print(f"     需进一步分析模型特性")
    print(f"  {'='*60}")

print(f"\n分析完成! 共分析 {len(analyzed_layers)} 层")
