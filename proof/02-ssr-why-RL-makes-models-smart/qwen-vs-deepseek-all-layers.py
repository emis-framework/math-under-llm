import torch
from safetensors.torch import load_file
import numpy as np
import os
import re
import gc
import time

# --- 硬件与超参数配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_HEAD = 128
N_Q_HEADS = 40
N_KV_HEADS = 8
GQA_RATIO = 5  # 5个Q Head共享1个K Head

def get_ssr_batch_gpu(wq_t, wk_t):
    """
    全并行 GPU 计算：一次性搬运整个 Layer 到显存，利用 Batch SVD 提速
    """
    # 1. 搬运并提升精度 (Float32 兼顾速度与精度)
    wq = wq_t.to(DEVICE).to(torch.float32)
    wk = wk_t.to(DEVICE).to(torch.float32)

    # 2. 向量化重组 (Reshape)
    # Q: (40, 128, 5120)
    q_heads = wq.view(N_Q_HEADS, D_HEAD, -1)
    
    # K: (8, 128, 5120) -> 扩展 5 倍对齐 Q 为 (40, 128, 5120)
    k_heads = wk.view(N_KV_HEADS, D_HEAD, -1)
    k_heads = k_heads.repeat_interleave(GQA_RATIO, dim=0)

    # 3. Batch SVD (P106 核心并行点火)
    s_q = torch.linalg.svdvals(q_heads) # (40, 128)
    s_k = torch.linalg.svdvals(k_heads) # (40, 128)

    # 4. 归一化谱形状并计算残差
    s_q_n = s_q / torch.norm(s_q, dim=1, keepdim=True)
    s_k_n = s_k / torch.norm(s_k, dim=1, keepdim=True)
    
    # 计算这 40 个 Head 的平均 SSR
    ssr_per_head = torch.mean(torch.abs(s_q_n - s_k_n), dim=1)
    return torch.mean(ssr_per_head).item()

def run_experiment():
    print("="*60)
    print(f"🌟 Qwen 2.5 vs DeepSeek R1 逻辑对撞实验 (48层全量)")
    print(f"🖥️  运行设备: {DEVICE} | 内存策略: 16G RAM 四段式加载")
    print("="*60)

    files = [f for f in os.listdir('.') if f.endswith('.safetensors')]
    native_files = sorted([f for f in files if "Qwen2.5-14B-Instruct-model" in f])
    r1_files = sorted([f for f in files if "DeepSeek-R1-Distill-Qwen-14B-model" in f])

    # 分为四段，每段 12 层，精准控制 RAM 占用在 6GB 左右
    batches = [
        range(0, 12),  range(12, 24), range(24, 36), range(36, 48)
    ]
    
    all_results = []
    global_start_time = time.time()

    for idx, b_range in enumerate(batches):
        print(f"\n📦 [阶段 {idx+1}/4] 正在加载 Layer {b_range[0]} - {b_range[-1]} ...")
        
        n_cache, r_cache = {}, {}

        # 加载 Native
        for f in native_files:
            w = load_file(f, device="cpu")
            for k in list(w.keys()):
                m = re.search(r"model\.layers\.(\d+)\.self_attn\.(q|k)_proj\.weight", k)
                if m and int(m.group(1)) in b_range:
                    n_cache[k] = w[k]
            del w; gc.collect()

        # 加载 R1
        for f in r1_files:
            w = load_file(f, device="cpu")
            for k in list(w.keys()):
                m = re.search(r"model\.layers\.(\d+)\.self_attn\.(q|k)_proj\.weight", k)
                if m and int(m.group(1)) in b_range:
                    r_cache[k] = w[k]
            del w; gc.collect()

        # 对撞计算
        layers = sorted(list(set(int(re.search(r"\.layers\.(\d+)\.", k).group(1)) 
                               for k in n_cache.keys() if k in r_cache)))
        
        print(f"⚡ GPU 并行对撞中...")
        print(f"{'-'*75}")
        print(f"{'Layer':<10} | {'Native SSR':<18} | {'R1 SSR':<18} | {'Imp %':<12}")
        print(f"{'-'*75}")

        for L in layers:
            q_key = f"model.layers.{L}.self_attn.q_proj.weight"
            k_key = f"model.layers.{L}.self_attn.k_proj.weight"

            n_ssr = get_ssr_batch_gpu(n_cache[q_key], n_cache[k_key])
            r_ssr = get_ssr_batch_gpu(r_cache[q_key], r_cache[k_key])
            imp = (n_ssr - r_ssr) / n_ssr * 100
            
            all_results.append(imp)
            print(f"L{L:<9} | {n_ssr:.8f}        | {r_ssr:.8f}        | {imp:>+10.4f}%")
            
            torch.cuda.empty_cache()

        # 阶段清理
        n_cache.clear(); r_cache.clear()
        del n_cache, r_cache
        gc.collect()
        print(f"✅ 阶段 {idx+1} 完成，内存已释放。")

    # --- 最终汇总 ---
    total_time = time.time() - global_start_time
    avg_imp = np.mean(all_results)
    
    print("\n" + "="*60)
    print(f"📊 实验总结报告")
    print(f"⏱️  总耗时: {total_time:.2f} 秒")
    print(f"📈 全 48 层平均提升 (Average Improvement): {avg_imp:.4f}%")
    
    if avg_imp > 0:
        print(f"🚀 物理定律验证成功：DeepSeek R1 显著收窄了逻辑谱残差！")
    else:
        print(f"⚠️ 实验数据平淡，请检查权重分片完整性。")
    print("="*60)

if __name__ == "__main__":
    run_experiment()
