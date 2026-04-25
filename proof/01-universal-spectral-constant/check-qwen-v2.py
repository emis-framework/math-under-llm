import torch
from safetensors.torch import load_file
import numpy as np
from scipy.stats import spearmanr
import os

# ---------------------------- 配置区 ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 模型超参数 (Qwen2.5-3B)
D_MODEL = 2048          # 隐藏层维度
D_HEAD = 128            # 每个头的维度
N_Q_HEADS = 16          # Query 头总数 (根据模型实际)
N_KV_HEADS = 2          # Key/Value 头总数 (Qwen2.5-3B 使用 GQA，2个 KV 头)
GROUP_SIZE = N_Q_HEADS // N_KV_HEADS   # 每组包含 8 个 Query 头

# 要分析的层范围 (Qwen2.5-3B 通常 28 层，可根据实际情况调整)
LAYERS_TO_ANALYZE = range(28)   # 0 ~ 27 层

# 权重文件路径 (第一个分片，需确保包含所有需要分析的层)
FILE_PATH = "qwen-model-00001-of-00002.safetensors"
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"请确认文件路径: {FILE_PATH}")

print(f"正在加载模型权重: {FILE_PATH} ...")
weights = load_file(FILE_PATH)

# ---------------------------- 工具函数 ----------------------------
def compute_pearson_corr_torch(x, y):
    """
    使用 PyTorch 在 GPU 上计算 Pearson 线性相关系数。
    x, y: 1D torch.Tensor
    """
    xm = x - x.mean()
    ym = y - y.mean()
    r_num = torch.dot(xm, ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    if r_den == 0:
        return 0.0
    return (r_num / r_den).item()

def analyze_layer(layer_idx):
    """
    分析指定层的所有 GQA 头对，计算两种相关系数。
    """
    # 构造该层的 Key 名称 (Qwen 命名规范)
    q_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
    k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
    
    if q_key not in weights or k_key not in weights:
        print(f"  跳过层 {layer_idx}: 权重键缺失 ({q_key} 或 {k_key})")
        return
    
    # 加载权重并移至 GPU (float32)
    wq = weights[q_key].to(torch.float32).to(DEVICE)
    wk = weights[k_key].to(torch.float32).to(DEVICE)
    
    print(f"\n========== Layer {layer_idx} ==========")
    print(f"  wq shape: {wq.shape}, wk shape: {wk.shape}")
    
    # 验证形状是否符合预期
    expected_q_shape = (N_Q_HEADS * D_HEAD, D_MODEL)
    expected_k_shape = (N_KV_HEADS * D_HEAD, D_MODEL)
    if wq.shape != expected_q_shape or wk.shape != expected_k_shape:
        print(f"  警告: 形状不符合预期，请检查超参数设置")
        print(f"  预期 Q: {expected_q_shape}, 实际 {wq.shape}")
        print(f"  预期 K: {expected_k_shape}, 实际 {wk.shape}")
    
    # 对每个 KV 头（共享 K 矩阵）循环
    for kv_h in range(N_KV_HEADS):
        # 提取当前 KV 头的 K 矩阵: [D_HEAD, D_MODEL]
        k_tensor = wk[kv_h * D_HEAD : (kv_h + 1) * D_HEAD, :]
        # SVD 分解，只取奇异值 (在 GPU 上计算)
        _, s_k, _ = torch.linalg.svd(k_tensor, full_matrices=False)
        
        # 该 KV 头对应的组内 Q 头索引范围
        start_q = kv_h * GROUP_SIZE
        end_q = start_q + GROUP_SIZE
        for q_offset in range(GROUP_SIZE):
            h_idx = start_q + q_offset
            # 提取 Q 头矩阵
            q_tensor = wq[h_idx * D_HEAD : (h_idx + 1) * D_HEAD, :]
            _, s_q, _ = torch.linalg.svd(q_tensor, full_matrices=False)
            
            # 截断奇异值向量使长度一致 (理论上长度均为 D_HEAD，但安全起见取最小)
            min_len = min(s_q.shape[0], s_k.shape[0])
            s_q_trunc = s_q[:min_len]
            s_k_trunc = s_k[:min_len]
            
            # ---------- 计算两种相关性 ----------
            # 1) Pearson 线性相关系数 (GPU)
            pearson_r = compute_pearson_corr_torch(s_q_trunc, s_k_trunc)
            # 2) Spearman 秩相关系数 (需要排序，移至 CPU 用 scipy)
            s_q_np = s_q_trunc.cpu().numpy()
            s_k_np = s_k_trunc.cpu().numpy()
            spearman_r, _ = spearmanr(s_q_np, s_k_np)
            
            # 打印结果 (仅显示前几个头作为样例，可自行注释限制)
            # 如果需要全部输出，删除 if 条件
            if h_idx < 16:   # 或者直接全部打印，删除此行
                print(f"  KV头 {kv_h} (组内Q偏移 {q_offset}) | Q头索引 {h_idx:2d}: "
                      f"Pearson = {pearson_r:+.4f}, Spearman = {spearman_r:+.4f}")

# ---------------------------- 主执行流程 ----------------------------
print(f"\n开始分析 {len(LAYERS_TO_ANALYZE)} 层，每层 {N_KV_HEADS} 个 KV 头 × {GROUP_SIZE} 个 Q 头")
for layer in LAYERS_TO_ANALYZE:
    try:
        analyze_layer(layer)
    except Exception as e:
        print(f"层 {layer} 处理出错: {e}")
