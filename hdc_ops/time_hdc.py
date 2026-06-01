"""
时间 HDC 引擎 — Elephant #14

Michael 的 4D 哲学：time 是最重要的维度。
这个模块 = HDC 大陆上的时间系统。

7 种时间编码 → 统一 API → 每个函数 = 一个具体的时间操作
"""

import numpy as np
from typing import Optional, List, Tuple


def clock_hv(D: int, frequency: float, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    大白话: 做一个"时钟"——一个以频率 f 旋转的 HV。
    在 Fourier 空间里 = e^{i·2πf·t} 的一个维度。

    场景: 每个维度 = 一个独立时钟。D 个维度 = D 个不同频率的时钟。
          快的捕捉毫秒，慢的捕捉世纪。

    Returns: Fourier HV e^{iθ} where θ ~ 2πf
    """
    if rng is None:
        rng = np.random.RandomState()
    phases = rng.uniform(0, 2 * np.pi, D) * frequency
    return np.exp(1j * phases).astype(np.complex64)


def encode_timepoint(t: float, clocks: np.ndarray) -> np.ndarray:
    """
    大白话: 把"时间点 t"编码成一个 HV。
    每个时钟根据自己的频率旋转——t 秒后，快时钟转了很多圈，慢时钟几乎没动。

    场景: "这个事件发生在 2024 年 3 月 15 日 14:30" → 一个 HV
          查询时找 cos 最近的 HV → 找到"同一天发生的事件"

    Args:
        t: 时间点（秒、天、年——取决于 clocks 的频率范围）
        clocks: (D,) Fourier HV — 从 clock_hv 创建的时钟数组
    Returns: Fourier HV
    """
    phases = np.angle(clocks) * t
    return np.exp(1j * phases)


def multi_scale_clock(D: int, scales: Optional[List[float]] = None, 
                      rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    大白话: 做一个多尺度时钟——同时有秒针、分针、时针。
    不同频率的时钟叠加在一起 → 一个 HV 同时编码毫秒和世纪。

    场景: 一个人的一生——婴儿期的记忆和今天的记忆在同一个 HV 里。
          cos 高的维度 = 同一年龄段。cos 低的维度 = 时间相隔远。

    Args:
        D: 维度
        scales: 时间尺度列表。默认 [1, 60, 3600, 86400, 31536000] 
                (秒、分、时、天、年)
    """
    if rng is None:
        rng = np.random.RandomState()
    if scales is None:
        scales = [1.0, 60.0, 3600.0, 86400.0, 31536000.0]  # sec, min, hr, day, yr
    
    D_per_scale = D // len(scales)
    clock_parts = []
    
    for i, scale in enumerate(scales):
        # Each scale gets its own frequency range within its D portion
        base_freq = 1.0 / scale
        freqs = base_freq * (1.0 + 0.1 * rng.randn(D_per_scale))
        phases = rng.uniform(0, 2 * np.pi, D_per_scale)
        # Higher frequency = bigger phase spread for same Δt
        h_scale = np.exp(1j * phases * freqs.reshape(-1, 1)).mean(axis=1)
        clock_parts.append(h_scale.ravel())
    
    h_multi = np.concatenate(clock_parts)[:D]
    return h_multi.astype(np.complex64)


def time_distance(h1: np.ndarray, h2: np.ndarray, clocks: np.ndarray) -> float:
    """
    大白话: 两个时间点的 HV 有多"远"？
    快时钟维度给出高精度（秒级差异），慢时钟维度给出大尺度（年级差异）。

    Returns: 0-1, 越接近 1 = 同一时间，越接近 0 = 时间相隔极远
    """
    phase_diff = np.angle(h1) - np.angle(h2)
    # Weighted by clock frequency: fast clocks contribute more to precision
    freqs = np.abs(np.angle(clocks))
    weights = freqs / (freqs.sum() + 1e-8)
    cos_diffs = np.cos(phase_diff)
    return float(np.dot(cos_diffs, weights))


def encode_event_sequence(events: List[np.ndarray], times: List[float],
                          clocks: np.ndarray) -> np.ndarray:
    """
    大白话: 一串事件按时间排列，每个事件跟时间点绑定。
    "2024 年的恐惧"、"2025 年的勇气"——两个事件的 HV 分别跟它们的时间绑在一起。

    场景: 个人知识库 时间线。一生的笔记→按时间编码→查询特定时间段的情绪变化。

    Args:
        events: List of event HVs (bipolar or Fourier)
        times: List of time points
        clocks: Clock HVs
    Returns: sequence HV (Fourier)
    """
    D = clocks.shape[0]
    h_seq = np.zeros(D, dtype=np.complex64)
    for event_hv, t in zip(events, times):
        time_hv = encode_timepoint(t, clocks)
        if not np.iscomplexobj(event_hv):
            # Convert bipolar to Fourier for bind
            event_hv = event_hv.astype(np.complex64)
        # Bind event with its time point
        h_bound = np.exp(1j * (np.angle(event_hv) + np.angle(time_hv)))
        h_seq += h_bound
    return h_seq / np.abs(h_seq).max()  # normalize


def temporal_bundle_split(h_seq: np.ndarray, clocks: np.ndarray, 
                          target_scale_idx: int, n_scales: int = 5) -> np.ndarray:
    """
    大白话: 从多尺度时间序列里提取特定尺度。
    "只看年级的规律，忽略秒级噪声" → 只保留慢时钟维度的信号。

    场景: 从 tokamak 数据里提取毫秒级涨落 vs 从气候数据里提取年级趋势。
          同一个 HV，不同尺度视角。
    """
    D = h_seq.shape[0]
    D_per_scale = D // n_scales
    start = target_scale_idx * D_per_scale
    end = start + D_per_scale
    return h_seq[start:end]


def time_permute(h: np.ndarray, dt: float, clocks: np.ndarray) -> np.ndarray:
    """
    大白话: 把 HV 沿时间轴移动 dt。
    不是离散移位（permute 的 k 步）——是连续滑动（fractional_bind）。

    场景: "从现在往未来推 3 天"、"从过去往回退 30 年"。
          不需要离散的时间步——时间是连续的。
    """
    phases = np.angle(h) + np.angle(clocks) * dt
    return np.exp(1j * phases).astype(np.complex64)
