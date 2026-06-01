"""
Paged SSD Hypervector Database — oMLX 启发

问题: D=1.3M × 1000万 docs × int8 = 13TB → RAM 装不下。
解法: SSD 作为虚拟内存 + 页交换 + LRU 缓存。

类比 oMLX:
  oMLX Paged KV Cache → HDC Paged HV Store
  KV page = 一批 token 的 KV → HV page = 一批文档的 HV
  TTFT 30s→5s → 首次查询 SSD 加载 → LRU 热页常驻 RAM

架构:
  SSD: 所有 HV 按页存储 (page_size=1000 docs/page)
  RAM: LRU cache (cache_size=100 pages = 10万 docs)
  Index: 每个 page 的 centroid HV (粗筛) → top-K pages → 加载 → 精搜
"""

import numpy as np
import os, struct, time, pickle
from typing import Optional, List, Tuple


class PagedHVDatabase:
    """
    大白话: 一个放在 SSD 上的超大规模 HV 数据库。
    10 亿文档？SSD 够大就行。查询时只加载相关的几页到 RAM。

    使用:
      db = PagedHVDatabase("hdc_index", D=1_300_000, page_size=1000)
      db.add_batch(docs_HV)  # 批量写入 SSD
      results = db.search(query_HV, k=10)  # 查询 — 自动换页

    成本模型 (M4 24GB, 1TB SSD):
      可存储: ~7000万 docs @ D=1.3M int8
      首次查询: ~50ms (SSD 加载 1 页)
      热查询: ~5ms (LRU cache hit)
    """

    def __init__(self, path: str, D: int = 1_000_000, page_size: int = 1000,
                 cache_pages: int = 100):
        self.path = path
        self.D = D
        self.page_size = page_size
        self.cache_pages = cache_pages

        # 每页的文件: {path}/page_00000.bin, page_00001.bin, ...
        # 每页的 centroid: (page_id, centroid_HV) → 存在 index.pkl
        os.makedirs(path, exist_ok=True)

        self._index = {}  # page_id → centroid_HV
        self._page_count = 0
        self._total_docs = 0

        # LRU cache: page_id → (HV_matrix, last_access_time)
        self._cache = {}
        self._cache_order = []  # LRU list — 最近用的在末尾

        # 加载已有索引
        self._load_index()

    def _page_path(self, page_id: int) -> str:
        return os.path.join(self.path, f"page_{page_id:06d}.bin")

    def _load_index(self):
        """从磁盘加载页索引。"""
        idx_path = os.path.join(self.path, "index.pkl")
        if os.path.exists(idx_path):
            with open(idx_path, 'rb') as f:
                data = pickle.load(f)
                self._index = data.get('index', {})
                self._page_count = data.get('page_count', 0)
                self._total_docs = data.get('total_docs', 0)

    def _save_index(self):
        """保存页索引到磁盘。"""
        idx_path = os.path.join(self.path, "index.pkl")
        with open(idx_path, 'wb') as f:
            pickle.dump({
                'index': self._index,
                'page_count': self._page_count,
                'total_docs': self._total_docs,
            }, f)

    def add_batch(self, hvs: np.ndarray) -> int:
        """批量写入文档。hvs: (N, D) bipolar int8。
        Returns: 新写入的文档数。
        """
        N = hvs.shape[0]
        if hvs.shape[1] != self.D:
            raise ValueError(f"Expected D={self.D}, got {hvs.shape[1]}")

        if N == 0:
            return 0

        hvs = hvs.astype(np.int8)
        added = 0

        # 写入一整页还是追加到最后一页（如果最后一页未满）
        current_page = self._page_count - 1 if self._page_count > 0 else 0
        current_path = self._page_path(current_page)
        current_count = 0

        if os.path.exists(current_path):
            current_count = os.path.getsize(current_path) // (self.D * 2)  # int8 + sign

        # 拆分成页
        pos = 0
        while pos < N:
            if current_count >= self.page_size or not os.path.exists(current_path):
                # 新页
                current_page = self._page_count
                self._page_count += 1
                current_path = self._page_path(current_page)
                current_count = 0

            # 写入当前页
            room = self.page_size - current_count
            chunk = min(room, N - pos)
            batch = hvs[pos:pos + chunk]

            with open(current_path, 'ab') as f:
                batch.tofile(f)

            # 更新 centroid
            if current_page in self._index:
                old_centroid = self._index[current_page]
                total = current_count + chunk
                old_weight = current_count / total if total > 0 else 0
                new_weight = chunk / total if total > 0 else 1
                self._index[current_page] = np.sign(
                    old_weight * old_centroid + new_weight * np.sign(np.sum(batch, axis=0))
                ).astype(np.int8)
            else:
                self._index[current_page] = np.sign(np.sum(batch, axis=0)).astype(np.int8)

            current_count += chunk
            pos += chunk
            added += chunk
            self._total_docs += chunk

        self._save_index()
        return added

    def _load_page(self, page_id: int) -> np.ndarray:
        """从 SSD 加载一页到 RAM + LRU cache。"""
        # 检查 cache
        if page_id in self._cache:
            # LRU: 移到末尾
            self._cache_order.remove(page_id)
            self._cache_order.append(page_id)
            return self._cache[page_id]

        # 从 SSD 加载
        path = self._page_path(page_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Page {page_id} not found: {path}")

        n_docs = os.path.getsize(path) // self.D
        hvs = np.fromfile(path, dtype=np.int8).reshape(n_docs, self.D)

        # 加入 LRU cache
        if len(self._cache) >= self.cache_pages:
            # 淘汰最老的
            evict_id = self._cache_order.pop(0)
            del self._cache[evict_id]

        self._cache[page_id] = hvs
        self._cache_order.append(page_id)
        return hvs

    def search(self, query_hv: np.ndarray, k: int = 10,
               coarse_pages: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        查询: 粗筛 top-K 页 → 加载 → 精搜 top-k。

        Args:
            query_hv: (D,) bipolar HV
            k: top-k 文档
            coarse_pages: 粗筛时考虑的页数
        Returns:
            (scores, global_doc_indices)
        """
        query_hv = query_hv.ravel().astype(np.float32)

        # Step 1: 粗筛 — 找跟 query 最像的 K 个 centroid (页)
        centroids = np.array([self._index[pid] for pid in range(self._page_count)
                             if pid in self._index], dtype=np.float32)
        page_ids = [pid for pid in range(self._page_count) if pid in self._index]

        if len(centroids) == 0:
            return np.array([]), np.array([])

        centroid_sims = centroids @ query_hv.astype(np.float32)
        top_pages = min(coarse_pages, len(centroids))
        top_page_idx = np.argpartition(-centroid_sims, top_pages - 1)[:top_pages]
        top_page_idx = top_page_idx[np.argsort(-centroid_sims[top_page_idx])]

        # Step 2: 精搜 — 加载 top-K 页，逐页搜索
        all_scores = []
        all_global_idx = []
        candidates_per_page = max(k * 2 // len(top_page_idx), k)  # 每页取多少候选

        for pi in top_page_idx:
            pid = page_ids[pi]
            page_hvs = self._load_page(pid).astype(np.float32)
            # 在页内搜索
            page_sims = page_hvs @ query_hv
            top_in_page = min(candidates_per_page, len(page_sims))
            page_top_idx = np.argpartition(-page_sims, top_in_page - 1)[:top_in_page]
            page_top_idx = page_top_idx[np.argsort(-page_sims[page_top_idx])]

            all_scores.extend(page_sims[page_top_idx])
            # 全局 doc index = page_id * page_size + doc_index_in_page
            all_global_idx.extend([pid * self.page_size + i for i in page_top_idx])

        # Step 3: 全局 top-k
        all_scores = np.array(all_scores)
        all_global_idx = np.array(all_global_idx)
        top_k = min(k, len(all_scores))
        global_top = np.argpartition(-all_scores, top_k - 1)[:top_k]
        global_top = global_top[np.argsort(-all_scores[global_top])]

        return all_scores[global_top], all_global_idx[global_top]

    @property
    def total_docs(self) -> int:
        return self._total_docs

    @property
    def page_count(self) -> int:
        return self._page_count

    def cache_stats(self) -> dict:
        """LRU cache 统计。"""
        return {
            'cache_size': len(self._cache),
            'cache_max': self.cache_pages,
            'cache_hit_rate': 'N/A — track externally',
            'total_docs': self._total_docs,
            'page_count': self._page_count,
            'disk_usage_gb': sum(
                os.path.getsize(self._page_path(pid))
                for pid in range(self._page_count)
                if os.path.exists(self._page_path(pid))
            ) / (1024 ** 3),
        }
