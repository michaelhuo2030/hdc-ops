"""
VSA 诊断面板 — Layer 3

基于 Dhayalkar 2025 的 VSA 指标体系，量化模型的"VSA-likeness"。
每个函数 = 一个具体的数学定义 + 一个具体的实验场景 + 大白话解释。

依赖: hdc_ops 算子库(08-infrastructure/hdc_ops/)
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import sys, os

# Ensure hdc_ops is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
try:
    from hdc_ops import bind, bundle, unbind, similarity, normalize
except ImportError:
    # Fallback inline implementations
    def bind(h1, h2): return h1 * h2
    def bundle(h_list): return np.sign(np.sum(np.stack(h_list, axis=0), axis=0))
    def unbind(hb, r): return hb * r
    def similarity(h1, h2): 
        return float(np.dot(normalize_impl(h1).ravel(), normalize_impl(h2).ravel()))
    def normalize_impl(h):
        n = np.linalg.norm(h, axis=-1, keepdims=True)
        return h / np.where(n == 0, 1.0, n)
    normalize = normalize_impl


# ═══════════════════════════════════════════════════
# 指标 1: Role-Filler Recoverability
# ═══════════════════════════════════════════════════

def recoverability(
    roles: np.ndarray,       # (N, D) — N 个角色 HV
    fillers: np.ndarray,     # (N, D) — N 个填充 HV
    query_role_idx: int,      # 查询哪个角色
    metric: str = "cosine",
) -> Dict:
    """
    大白话: N 个人(N 个角色)各自手里拿着一个东西(N 个填充)，全部叠在一个袋子里。
    你拿着其中一个人的钥匙(query_role)，问:"这个袋子里属于他的东西是什么？"
    你能恢复得多好？

    Dhayalkar: 这是衡量 attention 的 VSA-likeness 的核心度量。
    attention 中的 Q·K^T 相当于"选择哪个角色"——选出来后，V 就是"这个角色绑定的内容"。

    数学:
        z = bundle([bind(r_i, f_i) for i in range(N)])   # 叠加所有 bind
        f_recovered = unbind(z, r_query)                  # 用查询角色恢复
        recovery_score = similarity(f_recovered, f_true)  # 跟真实填充的相似度

    场景:
      - 测 FFN 的 VSA-likeness: 神经元当角色，权重当填充
      - 测 attention 的 VSA-likeness: key 当角色，value 当填充
      - 测任何 bind-based 系统的信号质量
    """
    r_query = roles[query_role_idx]
    f_true = fillers[query_role_idx]

    # 叠加所有 bind
    z = bundle([bind(roles[i], fillers[i]) for i in range(len(roles))])

    # 恢复
    f_recovered = unbind(z, r_query)

    # 度量
    recovery_score = similarity(f_recovered, f_true, metric=metric)

    # 对比基线的提升: 随机角色恢复(应该接近 0)
    random_role = np.sign(np.random.randn(*r_query.shape))
    f_random = unbind(z, random_role)
    random_score = similarity(f_random, f_true, metric=metric)

    return {
        "recovery_score": float(recovery_score),
        "random_baseline": float(random_score),
        "lift": float(recovery_score / (abs(random_score) + 1e-8)),
        "n_roles": len(roles),
    }


# ═══════════════════════════════════════════════════
# 指标 2: Interference Under Superposition
# ═══════════════════════════════════════════════════

def interference_curve(
    roles: np.ndarray,        # (total_N, D)
    fillers: np.ndarray,      # (total_N, D)
    n_range: List[int],        # 测试的 N 值列表，如 [1, 5, 10, 50, 100, 500]
    n_trials: int = 10,
    metric: str = "cosine",
) -> np.ndarray:
    """
    大白话: 袋子里放了 5 个东西时能恢复得怎么样？50 个呢？500 个呢？
    干扰多了以后，信号被稀释到什么程度？

    Dhayalkar: "interference under superposition"——这是衡量一个系统的
    "绑定容量"的核心指标。Transformer 的 attention 同时绑定 Q·K^T 个
    token 对的 key-value——这些绑定之间的干扰有多大？

    数学:
        对于每个 N:
          随机选 N 个角色+填充 → 叠加 → 随机抽取 query 恢复
          recovery_score(N) = mean over trials
        画出 recovery_score 对 N 的衰减曲线

    场景:
      - FFN-HDC: k 个激活神经元的干扰水平
      - RAG: bundle 里放了多少文档后检索质量下降
      - 飞轮: 蒸馏后模型能叠加多少层而不失真
    """
    total_N = len(roles)
    results = np.zeros((len(n_range), n_trials))

    for i, N in enumerate(n_range):
        if N > total_N:
            results[i, :] = np.nan
            continue
        for t in range(n_trials):
            idx = np.random.choice(total_N, N, replace=False)
            r = recoverability(roles[idx], fillers[idx], 
                             query_role_idx=np.random.randint(0, N), metric=metric)
            results[i, t] = r["recovery_score"]

    return results  # (len(n_range), n_trials)


# ═══════════════════════════════════════════════════
# 指标 3: Compositional Generalization
# ═══════════════════════════════════════════════════

def compositional_generalization(
    train_roles: np.ndarray,     # (N_train, D) — 训练时见过的角色
    train_fillers: np.ndarray,   # (N_train, D) — 训练时见过的填充
    test_roles: np.ndarray,      # (M_test, D) — 未见过的角色-填充组合
    test_fillers: np.ndarray,    # (M_test, D)
    metric: str = "cosine",
) -> Dict:
    """
    大白话: 你在训练时见过角色 A 拿苹果，角色 B 拿香蕉。
    现在给一个新组合——角色 A 拿香蕉——系统能正确解出来吗？

    这不是"记忆"——是"组合泛化"。系统是否真正理解了
    bind(role_i, filler_j) 的代数结构，而不只是记住了见过的组合。

    Dhayalkar: 这是衡量 VSA-likeness 的高级指标。
    真正的 VSA 系统应该对未见过的 role-filler 组合有泛化能力。
    因为 bind(r_A, f_banana) 和 bind(r_A, f_apple) 在代数上是
    相同操作的不同参数——如果系统学会了 bind 这个操作，泛化应该是自然的。

    场景:
      - 组合概念分类 (code AND error)
      - 跨语料库的概念探头—用语料 A 训练、在语料 B 上分类
      - 飞轮: 蒸馏后的 HDC 层能否泛化到新的输入分布
    """
    # 构建训练集的叠加
    if len(train_roles) < 5:
        return {"error": f"N_train={len(train_roles)} too small (need >=5)",
                "test_mean": 0, "test_std": 0, "train_mean": 0,
                "train_std": 0, "generalization_gap": 0, "n_test": 0}
    z_train = bundle([bind(train_roles[i], train_fillers[i]) 
                      for i in range(len(train_roles))])

    scores = []
    for i in range(len(test_roles)):
        z_test = bundle([z_train, bind(test_roles[i], test_fillers[i])])
        f_recovered = unbind(z_test, test_roles[i])
        s = similarity(f_recovered, test_fillers[i], metric=metric)
        scores.append(float(s))

    # 对比: 训练集内的恢复水平
    train_scores = []
    for i in range(min(len(train_roles), 50)):  # 抽 50 个
        f_rec = unbind(z_train, train_roles[i])
        train_scores.append(float(similarity(f_rec, train_fillers[i], metric=metric)))

    return {
        "test_mean": float(np.mean(scores)),
        "test_std": float(np.std(scores)),
        "train_mean": float(np.mean(train_scores)),
        "train_std": float(np.std(train_scores)),
        "generalization_gap": float(np.mean(train_scores) - np.mean(scores)),
        "n_test": len(scores),
    }


# ═══════════════════════════════════════════════════
# 指标 4: VSA-Likeness Score
# ═══════════════════════════════════════════════════

def vsa_likeness(
    roles: np.ndarray,
    fillers: np.ndarray,
    n_range: List[int] = [1, 5, 10, 50, 100],
    n_test_compositional: int = 20,
) -> Dict:
    """
    大白话: 把 recoverability、interference、compositional 合成一个分数。
    "这个系统的 VSA-likeness 是多少？"

    数字越大 → 越像真正的 VSA 系统。
    1.0 = 完美 VSA (理论上不存在)
    0.0 = 完全不像 VSA (随机)

    场景:
      - 诊断: 测每一层 Transformer 的 VSA-likeness
      - 对比: Qwen vs Gemma vs Mistral 的 VSA-likeness 曲线
      - 飞轮: 蒸馏前后 VSA-likeness 的变化—蒸馏让模型更 VSA 了吗？
      - 架构设计: MoE 的 VSA-likeness 低于 dense—需要不同的 HDC 策略
    """
    total_N = len(roles)
    recoveries = []
    
    # Component 1: Recovery @ N=10 (权重 0.4)
    n_base = min(10, total_N)
    for t in range(10):
        idx = np.random.choice(total_N, n_base, replace=False)
        r = recoverability(roles[idx], fillers[idx], 
                         query_role_idx=np.random.randint(0, n_base))
        recoveries.append(r["recovery_score"])
    recovery_score = float(np.mean(recoveries))

    # Component 2: Interference decay rate (权重 0.3)
    # SNR decay: recovery(N) ~ recovery(1) / sqrt(N)
    # Measure how close the decay follows 1/sqrt(N)
    ic = interference_curve(roles, fillers, n_range, n_trials=5)
    valid = ~np.isnan(ic).any(axis=1)
    if valid.sum() >= 2:
        n_vals = np.array(n_range)[valid]
        mean_rec = np.nanmean(ic, axis=1)[valid]
        log_n = np.log(n_vals)
        log_rec = np.log(mean_rec + 1e-8)
        # Slope of log-log: should be -0.5 for ideal VSA
        slope, _ = np.polyfit(log_n, log_rec, 1)
        interference_score = max(0.0, 1.0 - abs(slope + 0.5))  # 1.0 if slope = -0.5
    else:
        interference_score = 0.0

    # Component 3: Compositional generalization (权重 0.3)
    split = min(total_N // 2, 50)
    if split >= 10:
        train_idx = np.random.choice(total_N, split, replace=False)
        test_idx = np.array([i for i in range(total_N) if i not in train_idx])
        test_n = min(n_test_compositional, len(test_idx))
        test_idx = np.random.choice(test_idx, test_n, replace=False)
        
        # Create "unseen" combinations: take random role from train, random filler from test
        r_idx_t = np.random.choice(train_idx, test_n)
        f_idx_t = np.random.choice(test_idx, test_n)
        
        cg = compositional_generalization(
            roles[train_idx], fillers[train_idx],
            roles[r_idx_t], fillers[f_idx_t],
        )
        compositional_score = max(0.0, min(1.0, cg["test_mean"] / (cg["train_mean"] + 1e-8)))
    else:
        compositional_score = 0.5

    # Aggregate
    likeness = 0.4 * recovery_score + 0.3 * interference_score + 0.3 * compositional_score

    return {
        "vsa_likeness": float(likeness),
        "recovery_score": recovery_score,
        "interference_score": interference_score,
        "compositional_score": compositional_score,
        "n_roles_tested": total_N,
    }


# ═══════════════════════════════════════════════════
# 指标 5: Layer-Wise VSA Likeness Profile
# ═══════════════════════════════════════════════════

def layer_vsa_profile(
    layer_roles: Dict[int, Tuple[np.ndarray, np.ndarray]],
    # {layer_idx: (roles_array, fillers_array)}
) -> Dict[int, Dict]:
    """
    大白话: 把 Transformer 每一层都测一遍 VSA-likeness。
    画一条 VSA-likeness 对 layer 的曲线。

    预期 (基于已知实验):
      - L0 (embed): 高 likeness — 91× lift, token 分布统计 = VSA-like
      - L2-L6: 中高 — 局部上下文在 HDC 友好的深度
      - L11-L27: 下降 — 深层语义更正交，反 VSA
      - MoE: 整体更低 — expert routing = 竞争绑定通道

    场景:
      - 架构指纹: 每个模型一条 VSA likeness 曲线
      - 蒸馏决策: VSA likeness > 0.7 → HDC 蒸馏；< 0.3 → 保留 float
      - 论文: 架构敏感性的定量证据
    """
    profile = {}
    for layer_idx, (roles, fillers) in sorted(layer_roles.items()):
        profile[layer_idx] = vsa_likeness(roles, fillers)
    return profile
