"""
HDC 数据结构 — Layer 2

HVSet, HVMap, HVSequence — 三个标准容器。
不是"用 list 存 HV"——是"用代数操作实现集合/映射/序列"。

每个类 = 一个明确的抽象 + 数学定义 + 大白话解释 + 场景
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from hdc_ops import bundle, bind, unbind, permute, similarity, search, normalize
    from hdc_ops.encoding import simhash
except ImportError:
    def bundle(h_list, weights=None):
        s = np.stack(h_list, 0)
        if weights is not None:
            s = s * np.array(weights).reshape(-1, *([1]*(s.ndim-1)))
        return np.sign(s.sum(0))
    def bind(h1, h2): return h1 * h2
    def unbind(hb, r): return hb * r
    def permute(h, s=1): return np.roll(h, s, axis=-1)
    def similarity(h1, h2):
        n1 = h1/(np.linalg.norm(h1)+1e-8)
        n2 = h2/(np.linalg.norm(h2)+1e-8)
        return float(np.dot(n1.ravel(), n2.ravel()))


# ═══════════════════════════════════════════════════
# HVSet — 超向量集合
# ═══════════════════════════════════════════════════

class HVSet:
    """
    大白话: 一个袋子。往里面装东西，闻一闻袋子就知道这个东西在不在里面。

    数学: h_set = sign(Σ h_item_i)

    物理: 每个新元素叠进去，袋子"气味"改变一点——但以前的气味还在。

    场景:
      - RAG 文档集合: 所有 Mao 军事类文章的 HV 叠加
      - Motion: 所有 mocap frames 的动作库
      - 概念探头: 15 个例句叠成一个概念 HV
      - 异常检测: 正常数据的 HVSet vs 新数据点——cos 低 = 异常
      - 访问控制: 授权用户的 HVSet vs 新用户——cos 低 = 未授权访问
    """

    def __init__(self, D: int = 10000):
        self.D = D
        self._hv: Optional[np.ndarray] = None  # (D,) — 集合的 bundle
        self._items: List[np.ndarray] = []     # 独立存储所有元素
        self._count: int = 0

    def add(self, hv: np.ndarray) -> None:
        """叠加一个新元素。"""
        if self._hv is None:
            self._hv = hv.copy()
        else:
            self._hv = np.sign(self._hv + hv)
        self._items.append(hv.copy())
        self._count += 1

    def contains(self, hv: np.ndarray, tau: float = 0.5) -> bool:
        """这个元素在袋子里吗？cos > tau → 是。"""
        if self._hv is None:
            return False
        return similarity(hv, self._hv) > tau

    def remove(self, hv: np.ndarray) -> None:
        """从袋子里拿掉一个元素。近似操作——拿掉后袋子闻起来不太像它了。"""
        if self._hv is not None:
            self._hv = np.sign(self._hv - hv)
            self._items = [i for i in self._items if similarity(i, hv) < 0.9]

    @property
    def hv(self) -> Optional[np.ndarray]:
        return self._hv

    @property
    def count(self) -> int:
        return self._count


# ═══════════════════════════════════════════════════
# HVMap — 超向量映射
# ═══════════════════════════════════════════════════

class HVMap:
    """
    大白话: 键值对映射。把所有 (key, value) 用 bind 锁定后叠加在同一个 HV 里。
    用键去 unbind 就能拿出对应的值。

    数学: h_map = sign(Σ bind(key_i, value_i))
    SNR: ∝ 1/N。N 大时信号衰减——这是 bundle 的代数极限，不是 bug。
    大规模方案: 用 weighted_bundle 给重要键对更高权重，突破 1/N 墙。

    场景:
      - N≤50: HVMap 直接工作
      - N>50: 给高频/重要键对更高权重 (weighted_bundle)
      - N>500: 考虑分层——先 HVSet 粗筛候选，再 HVMap 精确匹配
    """

    def __init__(self, D=10000):
        self.D = D
        self._hv = None
        self._keys = []
        self._values = []
        self._count = 0

    def put(self, key, value, weight=1.0):
        """存一个键值对。weight 越高 → 这对在 bundle 里越突出。"""
        bound = bind(key, value)
        if self._hv is None:
            self._hv = weight * bound
        else:
            self._hv = np.sign(self._hv + weight * bound)
        self._keys.append(key.copy())
        self._values.append(value.copy())
        self._count += 1

    def get(self, key):
        """用键取出对应的值。unbind(h_map, key) → 恢复 value。"""
        if self._hv is None:
            raise KeyError("HVMap is empty")
        return unbind(self._hv, key)

    def contains_key(self, key, tau=0.3):
        """这个键在映射里吗？"""
        if self._hv is None:
            return False
        recovered = unbind(self._hv, key)
        for v in self._values:
            if similarity(recovered, v) > tau:
                return True
        return False

    @property
    def hv(self):
        return self._hv

    @property
    def count(self):
        return self._count


class HVSequence:
    """
    大白话: 一串事件按时间顺序叠在一起。每件事移不同的位——
    移 0 位是第一件，移 1 位是第二件……想找回第 k 件就反向移 k 位。

    数学: h_seq = sign(Σ ρ^k(h_k))

    物理: permute 是循环移位。移位量就是"时间步"。

    场景:
      - 西游记: 100 回的故事序列
      - 时间序列预测: 过去 10 帧
      - 对话历史: 前 5 轮
      - Motion: 连续动捕帧
      - Brain: 记忆回放序列
      - Tokamak: 等离子体涨落序列
    """

    def __init__(self, D: int = 10000):
        self.D = D
        self._hv: Optional[np.ndarray] = None
        self._items: List[np.ndarray] = []
        self._count: int = 0

    def append(self, hv: np.ndarray) -> None:
        """加一个新事件到序列末尾。D must exceed sequence length."""
        if self._count >= self.D - 1:
            raise OverflowError(f"HVSequence full: length {self._count} >= D {self.D}")
        shifted = permute(hv, shift=self._count)
        if self._hv is None:
            self._hv = shifted.copy()
        else:
            self._hv = np.sign(self._hv + shifted)
        self._items.append(hv.copy())
        self._count += 1

    def get(self, idx: int) -> np.ndarray:
        """取序列中的第 idx 个事件。idx=0 是第一件。"""
        if self._hv is None:
            raise IndexError("HVSequence is empty")
        if idx < 0 or idx >= self._count:
            raise IndexError(f"Index {idx} out of range [0, {self._count})")
        return permute(self._hv, shift=-idx)

    def match_prefix(self, query_seq: List[np.ndarray], tau: float = 0.5) -> int:
        """这个序列是否以 query_seq 开头？返回匹配的长度。"""
        for k, q_hv in enumerate(query_seq):
            recovered = permute(self._hv, shift=-k)
            if similarity(recovered, q_hv) < tau:
                return k
        return len(query_seq)

    @property
    def hv(self) -> Optional[np.ndarray]:
        return self._hv

    @property
    def count(self) -> int:
        return self._count

    def __len__(self) -> int:
        return self._count


# ═══════════════════════════════════════════════════
# HVGraph — 超向量图
# ═══════════════════════════════════════════════════

class HVGraph:
    """
    大白话: 一张关系网。节点是人，边是"认识"关系。
           想知道"A 认不认识 C"？A 和 C 之间有没有一条路径？

    数学: 每条边 bind(起点, 终点) → 叠加 → h_graph
          路径存在？unbind(unbind(h_graph, 起点), 终点)

    场景:
      - 知识图谱: bind(实体A, 关系) → h_edge
      - 社会网络: bind(人A, 人B) → h_connexion
      - 因果链: bind(原因, 结果) → h_causal_edge
      - 系统关系: bind(组件A, 组件B) → h_interaction
    """

    def __init__(self, D: int = 10000):
        self.D = D
        self._hv: Optional[np.ndarray] = None
        self._edges: List[Tuple[np.ndarray, np.ndarray]] = []
        self._count: int = 0

    def add_edge(self, source: np.ndarray, target: np.ndarray) -> None:
        """加一条有向边：source → target。"""
        edge = bind(source, target)
        if self._hv is None:
            self._hv = edge.copy()
        else:
            self._hv = np.sign(self._hv + edge)
        self._edges.append((source.copy(), target.copy()))
        self._count += 1

    def neighbors(self, node: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """找与 node 最相似的 k 个节点。返回 [(索引, similarity), ...]."""
        if self._hv is None:
            return []
        scores = []
        for source, target in self._edges:
            sim_s = similarity(node, source)
            sim_t = similarity(node, target)
            scores.append(max(sim_s, sim_t))
        top_k = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
        return [(i, s) for i, s in top_k]

    def path_exists(self, source: np.ndarray, target: np.ndarray, tau: float = 0.5) -> bool:
        """从 source 到 target 有路径吗？(one-hop only for now)"""
        if self._hv is None:
            return False
        # 直接边: bind(source, target) 在 h_graph 里
        edge_hv = bind(source, target)
        return similarity(edge_hv, self._hv) > tau

    @property
    def hv(self) -> Optional[np.ndarray]:
        return self._hv

    @property
    def count(self) -> int:
        return self._count


# Post-mortem fix: sign(0) → +1 for all bundle/sign operations
# Applied globally to __init__.py, structures.py, encoding.py
