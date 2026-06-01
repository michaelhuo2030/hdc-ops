"""
HDC 查询模板 — Layer 5

自然语言 → HDC 操作序列。
不是 DSL——是 10 个可组合的查询模式。
每个模式 = 输入(自然语言) → 输出(HDC 操作序列)。

基于 RAG 检索, 角色绑定检索, universal_concepts, narrative HDC 场景
"""

import numpy as np


# ═══════════════════════════════════════════════════
# 模式 1: 简单检索 — "找跟这个最像的"
# ═══════════════════════════════════════════════════

def query_similar(query_hv: np.ndarray, db: np.ndarray, k: int = 10):
    """大白话: "给我看跟这个最像的 k 个东西"。
    
    使用场景: "找类似于这篇文章的文档"、"这个姿态最接近哪个 mocap 帧"
    """
    from hdc_ops import search
    return search(query_hv, db, k=k)


# ═══════════════════════════════════════════════════
# 模式 2: AND 查询 — "A 而且 B"
# ═══════════════════════════════════════════════════

def query_and(concept_hvs: list, db: np.ndarray, k: int = 10):
    """大白话: "同时有关 A 和 B 的东西"。
    
    HDC: bundle(A_HV, B_HV) → 闻起来像 A 也像 B 的方向。
    使用场景: "军事 AND 外交"、"code AND error"(组合概念分类）
    """
    from hdc_ops import bundle, search
    query_hv = bundle(concept_hvs)
    return search(query_hv, db, k=k)


# ═══════════════════════════════════════════════════
# 模式 3: OR 查询 — "A 或者 B"
# ═══════════════════════════════════════════════════

def query_or(concept_hvs: list, db: np.ndarray, k: int = 10):
    """大白话: "有 A 或有 B 的东西"。
    
    HDC: 分别搜 A 和 B，合并去重。
    不能简单 bundle A 和 B——bundle 是 AND 语义，不是 OR。
    使用场景: "军事 或者 经济"
    """
    from hdc_ops import search
    all_indices = set()
    for hv in concept_hvs:
        idx, _ = search(hv, db, k=k)
        all_indices.update(idx.tolist())
    return sorted(all_indices)[:k]


# ═══════════════════════════════════════════════════
# 模式 4: 角色查询 — "在这个场景里，X 做了什么？"
# ═══════════════════════════════════════════════════

def query_role(scene_hv: np.ndarray, role_hv: np.ndarray, db: np.ndarray, k: int = 10):
    """大白话: "孙悟空在这个场景里做了什么？"
    
    HDC: unbind(scene_HV, 孙悟空角色_HV) → 孙悟空在这个场景里的动作方向。
    使用场景: role-binding 检索、故事查询、知识图谱
    """
    from hdc_ops import unbind, search
    action_hv = unbind(scene_hv, role_hv)
    return search(action_hv, db, k=k)


# ═══════════════════════════════════════════════════
# 模式 5: 时间查询 — "X 在时间 T 是怎样的？"
# ═══════════════════════════════════════════════════

def query_time(concept_hv: np.ndarray, time_step: int, seq_db: np.ndarray, k: int = 10):
    """大白话: "三年前关于恐惧的笔记是什么样的？"
    
    HDC: permute(概念_HV, -time_step) → 沿时间轴往回滑动 time_step 步。
    使用场景: 个人知识库 跨时间检索、narrative HDC、tokamak 时间序列
    """
    from hdc_ops import permute, search
    query_hv = permute(concept_hv, shift=-time_step)
    return search(query_hv, seq_db, k=k)


# ═══════════════════════════════════════════════════
# 模式 6: 反事实查询 — "如果没有 X，会怎样？"
# ═══════════════════════════════════════════════════

def query_counterfactual(system_hv: np.ndarray, remove_hv: np.ndarray):
    """大白话: "如果唐僧没念紧箍咒，悟空会不会走？"
    
    HDC: unbind(system_HV, 紧箍咒_HV) → 从系统里拿掉紧箍咒 → 看悟空的动作。
    使用场景: 叙事反事实、系统科学因果推理、MoE ablation
    """
    from hdc_ops import unbind
    return unbind(system_hv, remove_hv)


# ═══════════════════════════════════════════════════
# 模式 7: 跨领域查询 — "这个问题在其他领域怎么解决的？"
# ═══════════════════════════════════════════════════

def query_cross_domain(problem_hv: np.ndarray, domain_bundles: list, k: int = 5):
    """大白话: "我的问题跟哪个行业的底层结构最像？"
    
    HDC: cos(问题_HV, 行业_i_bundle) for i in 1..10000 → top-5
    使用场景: 他山之石、罕见病诊断、跨领域创新
    """
    from hdc_ops import compare_all
    domain_matrix = np.array([db for db in domain_bundles])
    sims = compare_all(problem_hv, domain_matrix)
    top_k = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top_k]


# ═══════════════════════════════════════════════════
# 模式 8: 概念探头 — "这个文本是不是关于 X？"
# ═══════════════════════════════════════════════════

def query_concept_probe(text_hv: np.ndarray, concept_hv: np.ndarray, tau: float = 0.5) -> bool:
    """大白话: "这段话是不是关于恐惧的？"
    
    HDC: cos(文本_HV, 恐惧_HV) > tau → 是。
    使用场景: LLM 安全监控、心理健康检测、概念分类
    """
    from hdc_ops import similarity
    return similarity(text_hv, concept_hv) > tau


# ═══════════════════════════════════════════════════
# 模式 9: 序列匹配 — "这段历史是不是以 X 开头的？"
# ═══════════════════════════════════════════════════

def query_sequence_match(seq_hv: np.ndarray, prefix_hvs: list, tau: float = 0.5) -> int:
    """大白话: "这个故事的下一段是什么？"
    
    HDC: permute(seq_HV, -k) for k=0,1,2... → cos with prefix[k]
    匹配到 prefix 全部匹配 → 下一段 = permute(seq_HV, -(len(prefix)+1))
    使用场景: 叙事续写、对话预测、motion 下一帧
    """
    from hdc_ops import permute, similarity
    for k, prefix_hv in enumerate(prefix_hvs):
        recovered = permute(seq_hv, shift=-k)
        if similarity(recovered, prefix_hv) < tau:
            return k  # 匹配到第 k-1 段
    return len(prefix_hvs)  # 全部匹配


# ═══════════════════════════════════════════════════
# 模式 10: 多编码投票 — "三种视角都同意吗？"
# ═══════════════════════════════════════════════════

def query_ensemble(query_x, db, encodings: list, k: int = 10):
    """大白话: "bipolar 说最像 A, Fourier 说最像 B, sparse 说最像 C——投票决定"。
    
    HDC: 每个编码独立编码 query → search → 各返回 top-k → 投票合并。
    使用场景: 多编码融合引擎(大象 #13)、高可靠性检索(医疗/金融)
    """
    from collections import Counter
    all_votes = []
    for enc_name, encode_fn in encodings:
        h_query = encode_fn(query_x)
        from hdc_ops import search
        idx, _ = search(h_query, db, k=k)
        all_votes.extend(idx.tolist())

    # 投票: 被最多编码选中的文档
    vote_counts = Counter(all_votes)
    return [doc_idx for doc_idx, _ in vote_counts.most_common(k)]
