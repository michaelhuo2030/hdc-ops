"""
HDC 算子库 — Layer 0: 编码方法
把现实世界的数据（浮点向量）翻译成 HDC 语言（超向量）

每种编码 = 一种翻译方式。不同的翻译方式适合不同的任务。
"""

import numpy as np
from typing import Optional


def bipolar(x: np.ndarray) -> np.ndarray:
    """
    最粗暴的翻译：只看正负号，不看大小。

    输入: x = [3.2, -0.1, 0.0, -5.7]
    输出: h = [+1,  -1,  +1,  -1]

    好处: 零成本，每个维度就是 1 bit。存储是 float32 的 1/32。
    坏处: 丢了所有幅度信息。3.2 和 0.001 的正号没区别。
    什么时候用: 快速检索、对质量要求不高的时候。
    """
    return np.sign(x)


_R_CACHE = {}  # (d, D, seed) → R matrix. Critical: all vectors must share same R.

def simhash(
    x: np.ndarray,
    D: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    随机投影 + 取符号。最常用的翻译方式。

    过程:
      1. 用一个固定的随机矩阵 R 把 x 从 d 维投到 D 维
      2. 取符号 → 每个维度变成 ±1

    直觉: 有点像"从 100 个不同角度拍一张照片，每张照片只记明暗（亮=+1，暗=-1）"。
          100 个角度够多的话，两张相似的照片拍出来的明暗序列也会相似。

    好处: 稳定、通用、有理论保证（维度集中定理）。
    坏处: 随机投影本身有一定噪声。
    什么时候用: 几乎所有场景的默认选择。
    """

    if rng is None:
        rng = np.random.default_rng()
    d = x.shape[-1]
    seed = id(rng)
    cache_key = (d, D, seed)
    if cache_key not in _R_CACHE:
        _R_CACHE[cache_key] = rng.normal(0, 1.0, (d, D))
    R = _R_CACHE[cache_key]
    # normalize to prevent overflow
    x_norm = x / (np.linalg.norm(x) + 1e-8)
    projection = x_norm @ R
    h = np.sign(projection); h[h == 0] = 1.0; return h


def sparse_ternary(
    x: np.ndarray,
    D: int = 10000,
    sparsity: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    只保留最重要的维度，其余扔掉。单条路径（向后兼容）。
    批量请用 sparse_ternary_batch（快 100-1000×）。
    """
    if rng is None:
        rng = np.random.default_rng()
    d = x.shape[-1]
    x_norm = x / (np.linalg.norm(x) + 1e-8)
    R = rng.normal(0, 1.0, (d, D))
    projection = x_norm @ R
    k = max(1, int((1 - sparsity) * D))
    h = np.zeros(D, dtype=np.float32)
    top_k_idx = np.argpartition(np.abs(projection), -k)[-k:]
    h[top_k_idx] = np.sign(projection[top_k_idx])
    return h


def sparse_ternary_batch(
    X: np.ndarray,
    D: int = 10000,
    sparsity: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    批量 sparse_ternary 编码。

    核心优化:
      1. 共享 R 矩阵（_R_CACHE）
      2. 一次 matmul: X @ R → (N, D)
      3. 批量 argpartition 取每行 top-k
      4. 向量化 scatter 填充

    速度: 对 4864×896 @ 896×500K，从 ~15min（逐个循环）→ < 30s（批量）。
    """
    if rng is None:
        rng = np.random.default_rng()
    N, d = X.shape
    seed = id(rng)
    cache_key = (d, D, seed, "sparse_ternary")
    if cache_key not in _R_CACHE:
        _R_CACHE[cache_key] = rng.normal(0, 1.0, (d, D))
    R = _R_CACHE[cache_key]

    # Normalize each row
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / (norms + 1e-8)

    projection = X_norm @ R  # (N, D)
    k = max(1, int((1 - sparsity) * D))

    H = np.zeros((N, D), dtype=np.float32)
    top_k_idx = np.argpartition(np.abs(projection), -k, axis=1)[:, -k:]

    # Vectorized fill
    row_idx = np.arange(N)[:, None]
    H[row_idx, top_k_idx] = np.sign(projection[row_idx, top_k_idx])
    return H


def fourier_encode(
    x: np.ndarray,
    D: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    傅里叶编码：每个维度是单位圆上的一个角度，不是 ±1。

    过程:
      1. 随机投影 R 把 x 投到 D 维
      2. 每个维度的值被映射到一个角度 θ_j = 2π · sigmoid(projection[j])
      3. 输出是复数 e^{iθ_j} —— 在单位圆上

    直觉: 普通的 bipolar 编码 = 每个维度只能指向"上"或"下"（±1）。
          傅里叶编码 = 每个维度可以指向 360° 任意方向。
          精度高得多，因为角度是连续的。

    关键优势: bind 操作在傅里叶空间里是角度相加，没有噪声。
             bipolar bind = XOR（会累积噪声）。
             傅里叶 bind = 角度相加（无损）。

    好处: 组合推理最强。适合需要反复 bind/unbind 的任务。
    坏处: 计算稍贵（复数乘法）。存储是 bipolar 的 2 倍。
    什么时候用: 组合推理、角色绑定、多步逻辑。
    """
    if rng is None:
        rng = np.random.default_rng()
    d = x.shape[-1]
    x_norm = x / (np.linalg.norm(x) + 1e-8)
    R = rng.normal(0, 1.0, (d, D))  # wider projection → uniform phases
    projection = x_norm @ R
    # Scale to [0, 2π] for uniform phase distribution
    proj_min, proj_max = projection.min(), projection.max()
    if proj_max > proj_min:
        phases = (projection - proj_min) / (proj_max - proj_min) * 2 * np.pi
    else:
        phases = np.zeros(D)
    return np.exp(1j * phases)


def multi_bit(
    x: np.ndarray,
    D: int = 10000,
    bits: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    保留幅度的编码：不止 ±1，可以有多个级别。

    过程:
      1. SimHash 投影到 D 维
      2. 按幅度分成 2^bits 个级别（如 2-bit = -2, -1, +1, +2）

    直觉: bipolar = 只能表达"同意"或"反对"（±1）。
          2-bit = 可以表达"强烈反对、反对、同意、强烈同意"。
          多出的信息来自原始向量的幅度。

    好处: 保留更多信息。在低 D 时可能缩小与 float32 的差距。
    坏处: 不稳定。效果依赖编码器的分布（在 Gemma 上退步）。
    什么时候用: 需要保留精细差异、但 bit 预算有限的时候。
    """
    if rng is None:
        rng = np.random.default_rng()
    d = x.shape[-1]
    x_norm = x / (np.linalg.norm(x) + 1e-8)
    R = rng.normal(0, 1.0, (d, D))  # scale=1.0 because x is normalized
    projection = x_norm @ R
    max_val = 2 ** (bits - 1)
    boundaries = np.linspace(-max_val + 1e-8, max_val - 1e-8, 2**bits - 1)
    quantized = np.digitize(projection, boundaries) - (2 ** (bits - 1))
    return quantized.astype(np.float32)
