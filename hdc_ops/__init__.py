"""
HDC 算子库 — 数学基板 + 高层算子
每个函数 = 明确定义的数学操作 + 什么时候用它

架构：
  Layer -1 (基板): bind, bundle, permute — 不可再分的原子操作
  Layer -1x (扩展基板): soft_bind, weighted_bundle — 基板的参数化变体
  Layer 0 (编码): encoding.py — 把现实数据翻译成 HV
  Layer 1 (算子): similarity, search, sequence — 基板的常用组合
"""

import numpy as np
from typing import Optional, Tuple, List, Union, Literal


# ═══════════════════════════════════════════════════
# Layer -1: 基板 — 不可再分的原子操作
# ═══════════════════════════════════════════════════

def bind(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """
    bind ⊗ : 元素乘。自逆：bind(bind(a,b), b) = a。

    大白话: 把两个东西锁在一起。钥匙是 role_HV —— 再用钥匙锁一次就开了。
    锁在一起的两个人，看起来跟谁都不像。

    场景:
      - "文档 X 的作者是 Michael" → bind(doc_hv, michael_hv)
      - "孙悟空在做 打" → bind(swk_hv, fight_hv)
      - 想找回"孙悟空的角色在做什么" → unbind(scene_hv, swk_role_hv)
      - 角色绑定检索（键值对场景）
   """
    return h1 * h2


def bundle(
    h_list: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    bundle ⊕ : 加权符号和。产生的 HV 跟所有输入都相似。

    大白话: 把一堆东西装进一个袋子。袋子闻起来像所有东西的混合气味。
    有权重时，重的东西气味更浓。

    场景:
      - "一个文集里所有「军事」主题的文章" → bundle(所有军事文章 HVs)
      - 长文档章节级 bundle（500-token chunks）
      - 动作库: 所有 mocap frames 的动作库
      - 概念探头: 15 个例句的 HV 叠成一个概念 HV
      - 类原型: C_k = sign(Σ h_{k,i})
    """
    stacked = np.stack(h_list, axis=0)
    if weights is not None:
        w = np.asarray(weights).reshape(-1, *([1] * (stacked.ndim - 1)))
        stacked = stacked * w
    return np.sign(np.sum(stacked, axis=0))


def permute(h: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    permute ρ : 沿最后一维循环移位。可逆：ρ^{-k} ○ ρ^k = id。

    大白话: 把向量循环移几位。第 1 件事不移，第 2 件移 1 位，第 3 件移 2 位。
    想找回第 k 件事，就反向移 k 位。

    场景:
      - "第一回 灵根育孕源流出, 第二回 悟彻菩提真妙理..." 
        → sequence([ch1_hv, ch2_hv, ch3_hv])
      - 时间序列: 传感器读数按时间步编码
      - ASR: 音频帧按时间顺序编码
      - 任何需要"顺序"的地方
    """
    return np.roll(h, shift, axis=-1)


# ═══════════════════════════════════════════════════
# Layer -1x: 扩展基板 — 基板的参数化变体
# ═══════════════════════════════════════════════════

def soft_bind(
    h1: np.ndarray,
    h2: np.ndarray,
    tau: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    soft_bind : bind(h1,h2) 和随机噪声之间的连续插值。

    大白话: 用橡皮筋绑，不是铁锁。τ=1 是铁锁，τ=0 是完全松的。
    Transformer 的 attention 就是用橡皮筋绑的——不是咔嚓锁死，是软的。

    场景:
      - Dhayalkar 2025: attention ≈ soft_unbind(QK^T)·V
      - 训练时: 用 τ 控制 binding 强度，让梯度能流过
      - 模拟"部分绑定": 角色和内容之间的关联不是 100% 确定的
      - 替换神经网络 FFN 层为 HDC 时，soft_bind 让训练稳定
    """
    if rng is None:
        rng = np.random.default_rng()
    bound = h1 * h2
    noise = np.sign(rng.normal(0, 1, bound.shape))
    return np.sign(tau * bound + (1 - tau) * noise)


def unbind(h_bound: np.ndarray, h_role: np.ndarray) -> np.ndarray:
    """
    unbind ≡ bind (自逆)。从 bind(content, role) 中恢复 content。

    大白话: 用钥匙开锁。bind(content, role) 再 bind 一次 role = content。

    场景:
      - retrieve_role(composite_hv, role_hv) 的底层实现
      - 任何需要"从这个 bind 里把内容拿出来"的操作
    """
    return h_bound * h_role


def weighted_bundle(
    pairs: List[Tuple[np.ndarray, float]],
) -> np.ndarray:
    """
    weighted_bundle : 带权重的 bundle。重要程度不同的东西，叠在一起时气味浓淡不同。

    大白话: 重的东西在袋子里气味更浓。你闻袋子时，更容易闻到它。

    场景:
      - RAG 检索: 文档与查询的相似度作为权重 → 更相关的文档在 bundle 里更突出
      - 概念探头: 更典型的例句给更高权重
      - 置信度加权: ML 模型的 softmax 输出作为 bundle 权重
      - 时间衰减: 越近的事件权重越大
    """
    h_list, weights = zip(*pairs)
    return bundle(list(h_list), weights=np.array(weights))


# ═══════════════════════════════════════════════════
# 工具函数 — 通用操作，不绑定特定 HDC 语义
# ═══════════════════════════════════════════════════

def normalize(h: np.ndarray) -> np.ndarray:
    """
    normalize: 拉到单位长度。比较时只看方向，不看长度。

    场景:
      - cosine similarity 的前置步骤
      - Fourier 向量: 保持在单位圆上
      - multi_bit 向量: 防止大幅度的维度 dominate similarity
    """
    norm = np.linalg.norm(h, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return h / norm


def threshold(h: np.ndarray, tau: float) -> np.ndarray:
    """
    threshold: 静音。把 |值| ≤ τ 的维度关掉。

    场景:
      - sparse_ternary 编码的基础: 只保留 top-k 幅度
      - 降噪: 去掉弱信号维度，减少 cosine 计算的噪声
      - 稀疏化任何连续 HV
    """
    mask = np.abs(h) > tau
    return h * mask


# ═══════════════════════════════════════════════════
# Layer 1: 算子 — 基板的常用组合模式
# ═══════════════════════════════════════════════════

def similarity(
    h1: np.ndarray,
    h2: np.ndarray,
    metric: Literal["cosine", "hamming", "dot", "overlap"] = "cosine",
) -> float:
    """
    similarity: 两个 HV 有多像。

    大白话: 闻一闻两个东西的气味有多接近。

    场景:
      - 检索: 查询 vs 文档
      - 分类: 测试样本 vs 概念 HV
      - 对齐检查: 同一个东西在 Qwen vs Gemma 里的表示有多像
      - 聚类: 两个 row 是否可以合并 (剪枝场景）
      - 重复检测: living_doc 里的 near-duplicate detection
    """
    if metric == "cosine":
        n1, n2 = normalize(h1), normalize(h2)
        return float(np.dot(n1.ravel(), n2.ravel()))
    elif metric == "hamming":
        D = h1.shape[-1]
        return float(np.sum(h1 != h2) / D)
    elif metric == "dot":
        return float(np.dot(h1.ravel(), h2.ravel()))
    elif metric == "overlap":
        return float(np.mean(np.sign(h1) == np.sign(h2)))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def search(
    h_query: np.ndarray,
    h_database: np.ndarray,
    k: int = 10,
    metric: Literal["cosine", "hamming", "dot"] = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    search: 在数据库里找最像的 k 个。

    大白话: 在一屋子气味里，找最接近目标气味的 k 个东西。

    场景:
      - RAG 检索: 用户查询 → 找最相关的文档
      - Motion: 当前姿态 → 找最相似的 mocap frame
      - Brain-as-HDC: 线索 → 检索最匹配的记忆
      - Universal Concepts: 跨语料库的概念桥接
    """
    if metric == "cosine":
        q = normalize(h_query).ravel()
        db = h_database.reshape(h_database.shape[0], -1)
        db_norm = np.linalg.norm(db, axis=-1, keepdims=True)
        db_norm = np.where(db_norm == 0, 1.0, db_norm)
        sims = np.dot(db / db_norm, q)
    elif metric == "hamming":
        D = h_database.shape[-1]
        sims = 1.0 - np.sum(h_database != h_query.ravel(), axis=-1) / D
    elif metric == "dot":
        q = h_query.ravel()
        db = h_database.reshape(h_database.shape[0], -1)
        sims = np.dot(db, q)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    top_k = min(k, len(sims))
    indices = np.argpartition(-sims, top_k - 1)[:top_k]
    indices = indices[np.argsort(-sims[indices])]
    return indices, sims[indices]


def set_membership(
    h_item: np.ndarray,
    h_set: np.ndarray,
    tau: float = 0.5,
    metric: Literal["cosine", "hamming"] = "cosine",
) -> bool:
    """
    set_membership: h_item 在 h_set 里面吗？

    大白话: 闻一下袋子——这个气味够不够像袋子里的东西？

    场景:
      - "这篇文章属于军事类吗？" → bundle 了所有军事文章
      - 异常检测: 新数据点跟已知集合够不够像
      - 访问控制: 用户的 HV 在授权集合里吗
    """
    return similarity(h_item, h_set, metric=metric) > tau


def retrieve_role(
    h_composite: np.ndarray,
    h_role: np.ndarray,
) -> np.ndarray:
    """
    retrieve_role: 从 bind(content, role) 里恢复 content。

    大白话: "孙悟空在这个场景里做了什么？" → 用"孙悟空"钥匙打开场景 HV。

    场景:
      - role-binding 检索: role-binding 检索
      - 故事查询: "第 3 回里孙悟空做了什么？"
      - 知识图谱: bind(entity, relation) → 查 entity 的所有 relation
    """
    return unbind(h_composite, h_role)


def sequence(
    h_list: List[np.ndarray],
    method: Literal["permute", "bind_chain"] = "permute",
) -> np.ndarray:
    """
    sequence: 把一串事件按时间顺序编码。

    大白话: 每件事移不同的位，叠在一起。
    "第一件是什么？" → 反向移 0 位就能恢复。

    场景:
      - 西游记: 100 回的故事序列
      - 时间序列预测: 过去 10 帧的 HV 序列
      - 对话历史: 前 5 轮对话的编码
      - ASR: 音素序列
    """
    if method == "permute":
        shifted = [permute(h, shift=k) for k, h in enumerate(h_list)]
        return bundle(shifted)
    elif method == "bind_chain":
        result = h_list[0]
        for h in h_list[1:]:
            result = bind(result, h)
        return result
    else:
        raise ValueError(f"Unknown method: {method}")


def compare_all(
    h_query: np.ndarray,
    h_database: np.ndarray,
    metric: Literal["cosine", "hamming", "dot"] = "cosine",
) -> np.ndarray:
    """
    compare_all: 查询 vs 整个数据库的完整相似度向量。

    大白话: 不是只要 top-k，是我要跟所有人的比较结果。

    场景:
      - 批量评估: 所有测试样本 vs 所有概念 HV
      - 聚类: 构建相似度矩阵
      - 异常检测: 看整个分布，不只 top-k
    """
    if metric == "cosine":
        q = normalize(h_query).ravel()
        db = h_database.reshape(h_database.shape[0], -1)
        db_norm = np.linalg.norm(db, axis=-1, keepdims=True)
        db_norm = np.where(db_norm == 0, 1.0, db_norm)
        return np.dot(db / db_norm, q)
    elif metric == "hamming":
        D = h_database.shape[-1]
        return 1.0 - np.sum(h_database != h_query.ravel(), axis=-1) / D
    elif metric == "dot":
        return np.dot(h_database.reshape(h_database.shape[0], -1), h_query.ravel())
    else:
        raise ValueError(f"Unknown metric: {metric}")
