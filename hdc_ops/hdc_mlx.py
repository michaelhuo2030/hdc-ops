#!/usr/bin/env python3
"""
MLX Metal GPU backend for HDC武器库 — Layer 0-3

每个函数 = numpy API 的 GPU 等价 + auto-fallback。
如果 MLX 不可用 → 回退到 numpy。
设计: import hdc_mlx as hm → hm.encode(x, W) (GPU if available)

2026-05-27 · M4 24GB验证 + Aux Mac 128GB优化
用法:
  from hdc_mlx import encode, search, is_available
  if is_available(): H = encode(x, W)  # GPU
"""

import numpy as np
import time

HAS_MLX = False
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    pass


def is_available() -> bool:
    """MLX Metal GPU 是否可用。"""
    return HAS_MLX


def device_info() -> str:
    """GPU 信息。"""
    if not HAS_MLX:
        return "CPU (numpy)"
    try:
        return str(mx.default_device())
    except:
        return "GPU (unknown)"


# ═══════════════════════════════════════════════════
# Layer 0: Encoding — GPU accelerated
# ═══════════════════════════════════════════════════

def encode_bipolar(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """HDC bipolar encoding: H = sign(x @ W.T)。
    x: (N, d) or (d,) → MLX handles both
    W: (D, d) projection matrix
    Returns: (N, D) bipolar {-1,+1} float32
    """
    if not HAS_MLX:
        return np.sign(x @ W.T if x.ndim == 2 else x @ W.T.reshape(-1))
    x_mx = mx.array(x) if not isinstance(x, mx.array) else x
    W_mx = mx.array(W) if not isinstance(W, mx.array) else W
    H = mx.sign(x_mx @ W_mx.T)
    mx.eval(H)
    return np.array(H)


def encode_batch(x: np.ndarray, W: np.ndarray, chunk_size: int = 5000) -> np.ndarray:
    """Chunked GPU encoding — avoids OOM on large N."""
    if not HAS_MLX:
        return np.sign(x @ W.T)
    N = x.shape[0]
    chunks = []
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        xi = mx.array(x[i:end])
        W_mx = mx.array(W) if i == 0 else chunks[0]  # reuse W
        if not isinstance(W_mx, mx.array):
            W_mx = mx.array(W)
        Hi = mx.sign(xi @ W_mx.T)
        mx.eval(Hi)
        chunks.append(np.array(Hi))
    return np.concatenate(chunks, axis=0)


# ═══════════════════════════════════════════════════
# Layer 1: Similarity & Search — GPU accelerated
# ═══════════════════════════════════════════════════

def cosine_similarity(H_docs: np.ndarray, q: np.ndarray) -> np.ndarray:
    """全库 cosine 相似度。H_docs: (N, D), q: (D,) → (N,)。"""
    if not HAS_MLX:
        qn = q / (np.linalg.norm(q) + 1e-8)
        Hn = H_docs / (np.linalg.norm(H_docs, axis=1, keepdims=True) + 1e-8)
        return Hn @ qn
    H = mx.array(H_docs) if not isinstance(H_docs, mx.array) else H_docs
    q_mx = mx.array(q) if not isinstance(q, mx.array) else q
    # Normalize on GPU
    q_norm = q_mx / (mx.linalg.norm(q_mx) + 1e-8)
    H_norm = H / (mx.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    sims = H_norm @ q_norm
    mx.eval(sims)
    return np.array(sims)


def search_topk(H_docs: np.ndarray, q: np.ndarray, k: int = 10):
    """Top-k 搜索。Returns: (scores, indices)。"""
    if not HAS_MLX:
        sims = cosine_similarity(H_docs, q)
        idx = np.argpartition(-sims, min(k, len(sims)) - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return sims[idx], idx
    H = mx.array(H_docs) if not isinstance(H_docs, mx.array) else H_docs
    q_mx = mx.array(q) if not isinstance(q, mx.array) else q
    q_norm = q_mx / (mx.linalg.norm(q_mx) + 1e-8)
    H_norm = H / (mx.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    sims = H_norm @ q_norm
    topk = min(k, H_docs.shape[0])
    idx = mx.argsort(-sims)[:topk]
    vals = sims[idx]
    mx.eval(vals, idx)
    return np.array(vals), np.array(idx)


# ═══════════════════════════════════════════════════
# Layer 2: HVSet batch operations — GPU accelerated
# ═══════════════════════════════════════════════════

def bundle_gpu(hvs: np.ndarray) -> np.ndarray:
    """GPU 批量 bundle: h_bundle = sign(sum(H, axis=0))。"""
    if not HAS_MLX:
        return np.sign(np.sum(hvs, axis=0))
    H = mx.array(hvs) if not isinstance(hvs, mx.array) else hvs
    result = mx.sign(mx.sum(H, axis=0))
    mx.eval(result)
    return np.array(result)


def batch_search(H_db: np.ndarray, queries: np.ndarray, k: int = 10):
    """批量搜索: 多个 query 同时搜。queries: (n_queries, D)。"""
    if not HAS_MLX:
        results = []
        for i in range(queries.shape[0]):
            _, idx = search_topk(H_db, queries[i], k)
            results.append(idx)
        return np.array(results)
    H = mx.array(H_db) if not isinstance(H_db, mx.array) else H_db
    Q = mx.array(queries) if not isinstance(queries, mx.array) else queries
    H_norm = H / (mx.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    Q_norm = Q / (mx.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    sims = H_norm @ Q_norm.T  # (N, nq)
    idx = mx.argsort(-sims, axis=0)[:k, :].T  # (nq, k)
    mx.eval(idx)
    return np.array(idx)


# ═══════════════════════════════════════════════════
# Layer 3: VSA Diagnostics — GPU accelerated
# ═══════════════════════════════════════════════════

def recoverability_gpu(roles: np.ndarray, fillers: np.ndarray, query_idx: int):
    """GPU recoverability: z = sign(Σ bind(r_i, f_i)) → unbind with r_query。
    roles: (N, D), fillers: (N, D), D must be large for meaningful results.
    """
    if not HAS_MLX:
        z = np.sign(np.sum(roles * fillers, axis=0))
        return z * roles[query_idx]
    R = mx.array(roles) if not isinstance(roles, mx.array) else roles
    F = mx.array(fillers) if not isinstance(fillers, mx.array) else fillers
    # bind = element-wise multiply, bundle = sign(sum)
    z = mx.sign(mx.sum(R * F, axis=0))
    recovered = z * R[query_idx]  # unbind = multiply again
    mx.eval(recovered)
    return np.array(recovered)


# ═══════════════════════════════════════════════════
# Benchmark — 测本机性能
# ═══════════════════════════════════════════════════

def bench_encode(N: int = 5000, D: int = 50000, d: int = 896, n_trials: int = 3):
    """单次编码 benchmark。"""
    x = np.random.randn(N, d).astype(np.float32)
    W = np.random.randn(D, d).astype(np.float32)
    
    # Warmup
    if HAS_MLX: encode_bipolar(x[:100], W)
    
    t0 = time.perf_counter()
    for _ in range(n_trials):
        H = encode_bipolar(x, W) if not HAS_MLX else encode_batch(x, W)
    dt = time.perf_counter() - t0
    return dt / n_trials


def bench_search(N: int = 5000, D: int = 50000, k: int = 10, n_trials: int = 3):
    """单次搜索 benchmark。"""
    H = np.random.randn(N, D).astype(np.float32)
    q = np.random.randn(D).astype(np.float32)
    
    if HAS_MLX:
        _, _ = search_topk(H[:100], q, k)  # warmup
    
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _, _ = search_topk(H, q, k)
    dt = time.perf_counter() - t0
    return dt / n_trials


def quick_benchmark():
    """快速对比: numpy vs MLX (if available)。"""
    print(f"Device: {device_info()}")
    print(f"{'Op':<20} {'NumPy':>10} {'MLX':>10} {'Speedup':>10}")
    print("-" * 50)
    
    configs = [(1000, 10000), (5000, 50000), (10000, 100000)]
    for N, D in configs:
        t_np = bench_encode(N, D)
        if HAS_MLX:
            t_mlx = bench_encode(N, D)
            sp = t_np / t_mlx if t_mlx > 0 else float('inf')
            print(f"encode N={N:<5} D={D:<6} {t_np*1000:>8.0f}ms {t_mlx*1000:>8.0f}ms {sp:>9.0f}x")
        else:
            print(f"encode N={N:<5} D={D:<6} {t_np*1000:>8.0f}ms {'---':>10} {'---':>10}")
    
    for N, D in configs:
        t_np_s = bench_search(N, D)
        if HAS_MLX:
            t_mlx_s = bench_search(N, D)
            sp_s = t_np_s / t_mlx_s if t_mlx_s > 0 else float('inf')
            print(f"search N={N:<5} D={D:<6} {t_np_s*1000:>8.0f}ms {t_mlx_s*1000:>8.0f}ms {sp_s:>9.0f}x")
        else:
            print(f"search N={N:<5} D={D:<6} {t_np_s*1000:>8.0f}ms {'---':>10} {'---':>10}")


if __name__ == '__main__':
    quick_benchmark()
