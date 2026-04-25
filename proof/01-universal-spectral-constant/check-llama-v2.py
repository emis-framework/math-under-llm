import torch
from safetensors.torch import load_file
import numpy as np
from scipy.stats import spearmanr

# ---------------------------- 配置区 ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 模型超参数（请根据你的实际模型修改）
N_Q_HEADS = 32      # Q 头的总数
N_KV_HEADS = 8      # K/V 头的总数 (GQA 分组数)
D_MODEL = 4096      # 隐藏层维度
D_HEAD = 128        # 每个头的维度
GROUP_SIZE = N_Q_HEADS // N_KV_HEADS   # 每组包含的 Q 头数量 (此处为 4)

# 要分析的层范围（如果只想分析某一层，可改为 [7] 这样的列表）
#LAYERS_TO_ANALYZE = range(32)   # 分析 0~31 层，可根据需要修改
LAYERS_TO_ANALYZE = range(8)   # 分析 0~8 层，因为只有模型第一个分片

# 加载 safetensors 文件
FILE_PATH = "llama-3-model-00001-of-00004.safetensors"
weights = load_file(FILE_PATH)

# ---------------------------- 自动识别 Key 模板 ----------------------------
all_keys = list(weights.keys())

# 尝试匹配某一层的 Q 权重键名（以第 4 层为模板）
possible_q_keys = [k for k in all_keys if "layers.4." in k and ("wq" in k or "q_proj" in k)]
if not possible_q_keys:
    print("未找到包含 'layers.4.' 和 'wq' 或 'q_proj' 的键。请检查前20个键名：")
    print(all_keys[:20])
    exit()

template_q = possible_q_keys[0].replace("4", "{L}")
# 生成对应的 K 权重模板（将 q 替换为 k）
template_k = template_q.replace("q_proj", "k_proj").replace("wq", "wk")
print(f"识别到的键名模板:\n  Q: {template_q}\n  K: {template_k}")

# ---------------------------- 定义工具函数 ----------------------------
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
    # 加载该层的 Q 和 K 权重矩阵 (形状均为 [num_heads * head_dim, d_model])
    wq = weights[template_q.format(L=layer_idx)]
    wk = weights[template_k.format(L=layer_idx)]
    print(f"\n========== Layer {layer_idx} ==========")
    print(f"  wq shape: {wq.shape}, wk shape: {wk.shape}")
    
    # 对每个 KV 头（共享 K 矩阵）循环
    for kv_h in range(N_KV_HEADS):
        # 提取当前 KV 头的 K 矩阵: [head_dim, d_model]
        k_tensor = wk[kv_h * D_HEAD : (kv_h + 1) * D_HEAD, :].to(torch.float32).to(DEVICE)
        # SVD 分解，只取奇异值 s_k (在 GPU 上计算)
        _, s_k, _ = torch.linalg.svd(k_tensor, full_matrices=False)

        
        # 该 KV 头对应的组内 Q 头索引范围
        start_q = kv_h * GROUP_SIZE
        end_q = start_q + GROUP_SIZE
        for q_offset in range(GROUP_SIZE):
            h_idx = start_q + q_offset
            # 提取 Q 头矩阵
            q_tensor = wq[h_idx * D_HEAD : (h_idx + 1) * D_HEAD, :].to(torch.float32).to(DEVICE)
            _, s_q, _ = torch.linalg.svd(q_tensor, full_matrices=False)
            
            # 截断奇异值向量使长度一致（理论上 s_q 和 s_k 长度均为 head_dim=128）
            # 但为了安全，取两者最小长度
            min_len = min(s_q.shape[0], s_k.shape[0])
            s_q_trunc = s_q[:min_len]
            s_k_trunc = s_k[:min_len]
            
            # ---------- 计算两种相关性 ----------
            # 1) Pearson 线性相关系数 (在 GPU 上计算，然后取标量)
            pearson_r = compute_pearson_corr_torch(s_q_trunc, s_k_trunc)
            
            # 2) Spearman 秩相关系数 (需要排序，移到 CPU 用 scipy 计算)
            # 注意：scipy.stats.spearmanr 期望一维数组，np.asarray 会自动处理
            s_q_np = s_q_trunc.cpu().numpy()
            s_k_np = s_k_trunc.cpu().numpy()
            spearman_r, _ = spearmanr(s_q_np, s_k_np)
            
            # 打印结果
            print(f"  KV头 {kv_h} (组内Q偏移 {q_offset}) | Q头索引 {h_idx:2d}: "
                  f"Pearson = {pearson_r:+.4f}, Spearman = {spearman_r:+.4f}")

# ---------------------------- 主执行流程 ----------------------------
print(f"\n开始分析 {len(LAYERS_TO_ANALYZE)} 层，每层 {N_KV_HEADS} 个 KV 头 × {GROUP_SIZE} 个 Q 头")
for layer in LAYERS_TO_ANALYZE:
    try:
        analyze_layer(layer)
    except KeyError as e:
        print(f"层 {layer} 键不存在，跳过。缺失: {e}")
    except Exception as e:
        print(f"层 {layer} 处理出错: {e}")
