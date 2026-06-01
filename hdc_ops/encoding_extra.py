"""
HDC 算子库 — Layer 0 补充编码 (Missing Encodings)

Walsh-Hadamard, qFHRR, fractional_bind, MAP,
permutation_sequence, random_fourier_features

基于 Kimi 51 论文 + 文献考古
"""

import numpy as np
from typing import Optional, List
from scipy.linalg import hadamard as scipy_hadamard
from scipy.fft import fft


# ═══════════════════════════════════════════════════
# 1. Walsh-Hadamard 编码
# ═══════════════════════════════════════════════════

def walsh_hadamard(x: np.ndarray, D: Optional[int] = None) -> np.ndarray:
    """
    大白话: 随机投影的更快替代品。不用随机矩阵——用固定的 Hadamard 矩阵。
    Hadamard = 全部 ±1 的正交矩阵，乘起来跟随机投影类似但可以用快速变换。

    优势: O(d log d) vs SimHash 的 O(D·d)。不需要存 R 矩阵。不需要随机种子。
    劣势: D 必须是 2 的幂。对某些数据可能不如随机投影（因为 Hadamard 是确定的）。
    什么时候用: 大量 encoding 时替代 SimHash 节省时间。嵌入式场景（no_std，不需要存储 R）。

    来源: Kimi 发现 — arXiv:2410.22669 Walsh-Hadamard VSA
    """
    d = len(x)
    if D is None:
        # 找到 ≥ 4d 的最小 2 的幂
        D = 1
        while D < 4 * d:
            D *= 2

    # Pad x to D dimensions
    padded = np.zeros(D)
    padded[:d] = x

    # Fast Walsh-Hadamard Transform (FWHT)
    # H_D · x_padded → sign
    h = padded.copy()
    step = 1
    while step < D:
        for i in range(0, D, step * 2):
            for j in range(step):
                u = h[i + j]
                v = h[i + j + step]
                h[i + j] = u + v
                h[i + j + step] = u - v
        step *= 2

    return np.sign(h).astype(np.float32)


# ═══════════════════════════════════════════════════
# 2. qFHRR — 量化傅里叶全息约简表示
# ═══════════════════════════════════════════════════

def qfhhr(x: np.ndarray, D: int = 5000, bits: int = 8) -> np.ndarray:
    """
    大白话: Fourier HRR 的量化版。相位角不连续——被量化到 2^bits 个级别。
    精度比 Fourier 低但存储小、计算快。

    bind 操作: 角度相加 → 取最近量化级别
    优势: 整数角度 = int8 存储。bind = 整数加法（比 float 复数乘更快）。
    劣势: 量化噪声。
    什么时候用: 需要大量 bind/unbind 的任务（叙事 HDC、知识图谱推理）。
                在芯片上 bind = 整数加法（比 XOR 慢但保留更多信息）。

    来源: Kimi 发现 — arXiv:2604.25939
    """
    d = len(x)
    R = np.random.default_rng().normal(0, 1 / np.sqrt(d), (d, D))
    projection = x @ R

    # 量化相位: 0 to 2π → 0 to 2^bits - 1
    phases_continuous = (np.angle(np.exp(1j * projection)) + np.pi) / (2 * np.pi)
    quantized = np.floor(phases_continuous * (2 ** bits)).astype(np.uint16)
    return quantized


# ═══════════════════════════════════════════════════
# 3. Fractional Binding — 连续绑定
# ═══════════════════════════════════════════════════

def fractional_bind(h: np.ndarray, alpha: float) -> np.ndarray:
    """
    大白话: 不是"咔嚓锁死"(bind = XOR)——是"拧螺丝拧到 α 圈"(fractional_bind)。
    α=1 = 完整 bind。α=0.5 = 半 bind。α=0 = 不 bind。

    物理意义: 沿时间轴连续滑动。
    h(t) = fractional_bind(h_0, t) → t 是连续参数，不是离散步。

    场景:
      - 时间编码: h(t) = fractional_bind(h_0, t) — 连续时间，不是离散帧
      - Motion: 关节角在两个姿态之间连续插值
      - soft_bind 的底层实现: soft_bind(h1, h2, τ) ≈ bind(h1, fractional_bind(h2, τ))
      - 连续变换: 一个对象从状态 A 过渡到状态 B, α 是过渡进度

    实现: 在 Fourier 空间里——相位乘以 α。bipolar 空间里没法做。
          必须在 Fourier 空间做，再转回 bipolar（如果需要）。

    来源: Frady/Voelker 2022
    """
    if np.iscomplexobj(h):
        # Fourier space: fractional_bind(h, α) = h^{(1-α)}
        # α=0 → identity (cos=1 with original)
        # α=1 → full bind with random (cos≈0 with original)
        # Phases shrink toward zero as α→1
        return np.exp(1j * np.angle(h) * (1.0 - alpha))
    else:
        # Bipolar space: fractional_bind 不可行。
        # 随机翻转 bit 不是 fractional bind — 是 stochastic noise。
        # 必须在 Fourier 空间操作，或先转 Fourier 再转回。
        raise ValueError(
            "fractional_bind requires Fourier (complex) HV. "
            "Bipolar approximation is mathematically invalid — "
            "random bit flip ≠ continuous phase shift. "
            "Use fourier_encode() first, or apply fractional_bind "
            "in Fourier space then convert back with sign()."
        )


# ═══════════════════════════════════════════════════
# 4. MAP 编码 (Multiply-Add-Permute)
# ═══════════════════════════════════════════════════

def map_encode(x: np.ndarray, D: int = 10000) -> np.ndarray:
    """
    大白话: 实值 VSA——不量化到 ±1，保留浮点幅度。
    bind = 卷积(不是 XOR)。bundle = 实值相加(不是 sign)。

    为什么不用 bipolar: MAP 保留幅度，bind 在傅里叶空间是卷积——不累积噪声。

    场景:
      - 需要保留精确幅度的任务
      - 科学测量: 传感器的浮点读数不能量化
      - 作为对比基线: "bipolar 丢了多少精度？"

    来源: Gayler 2003 MAP VSA
    """
    d = len(x)
    D = max(D, 2 * d)
    h = np.zeros(D)
    h[:d] = x
    return h.astype(np.float32)


# ═══════════════════════════════════════════════════
# 5. Permutation Sequence 编码
# ═══════════════════════════════════════════════════

def permutation_sequence(h_list: List[np.ndarray]) -> np.ndarray:
    """
    大白话: 最朴素的序列编码——每件事移不同的位，叠在一起。
    跟 HVSequence 类一样——这里是函数版。

    想找回第 k 件事: np.roll(h_seq, -k)。在 ±1 空间里，permute 是可逆的。
    """
    D = h_list[0].shape[-1]
    h_seq = np.zeros(D)
    for k, h in enumerate(h_list):
        h_seq += np.roll(h, k, axis=-1)
    return np.sign(h_seq).astype(np.float32)


# ═══════════════════════════════════════════════════
# 6. Random Fourier Features 编码
# ═══════════════════════════════════════════════════

def random_fourier_features(x: np.ndarray, D: int = 5000, gamma: float = 1.0) -> np.ndarray:
    """
    大白话: 用 cos/sin 把输入投影到高维——跟 RBF kernel 的近似等价。
    跟 SimHash 有点类似，但不是取 sign——是 cos/sin。

    优势: 有理论保证(Rahimi & Recht 2007)——RFF 可以任意精度逼近任何平移不变核。
    劣势: D 必须偶数(cos + sin 对)。比 SimHash 需要更多 D 来达到相同精度。

    场景:
      - 核方法: 需要 RBF kernel 但不要 O(N^2) 的核矩阵
      - 大 D 下的稳健编码: RFF 在 D → ∞ 时收敛到精确核

    来源: Rahimi & Recht 2007 Random Features for Large-Scale Kernel Machines
    """
    d = len(x)
    D_half = D // 2
    rng = np.random.default_rng()
    W = rng.normal(0, 2 * gamma, (D_half, d))
    b = rng.uniform(0, 2 * np.pi, D_half)

    projection = W @ x + b
    features = np.sqrt(2.0 / D) * np.concatenate([np.cos(projection), np.sin(projection)])
    return features.astype(np.float32)
