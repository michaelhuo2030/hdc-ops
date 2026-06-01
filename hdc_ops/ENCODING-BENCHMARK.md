# HDC 编码性能矩阵 — Layer 0 系统归档

> 2026-05-27  
> 每种编码方法 = 一种"翻译方式"。不同的翻译在同一个任务上表现不同。  
> 本文档是对编码方法的系统性归档：每种编码在什么 D、什么场景、什么指标下表现如何。

---

## 编码方法速查

| 编码 | 输出类型 | 维度范围 | 存储/D | bind 方式 | 组合推理 |
|------|---------|---------|--------|---------|---------|
| `bipolar` | {-1,+1}^d | =d_model | 1 bit | XOR (有噪声累积) | 弱 |
| `simhash` | {-1,+1}^D | 1K-3M | 1 bit | XOR (有噪声累积) | 中 |
| `sparse_ternary` | {-1,0,+1}^D | 10K-1.3M | ~0.05 bit (稀疏) | XOR (稀疏) | 中 |
| `fourier_encode` | {e^{iθ}}^D | 1K-20K | 2×32 bit (复数) | 角度相加 (零噪声!) | **最强** |
| `multi_bit` | {-n,...,+n}^D | 1K-100K | 2-4 bit | 逐元素乘 (有量化噪声) | 中 |

---

## 检索性能矩阵（实测数据）

### SciFact (5,183 docs × 300 queries, bge-small-en, D=10K)

| 编码 | R@5 | vs dense | 存储节省 |
|------|-----|---------|---------|
| dense fp32 | 0.765 | — | 1× |
| bipolar | 0.754 | -0.011 | 32× |
| multi_bit 2-bit | 0.763 | -0.002 | 16× |
| multi_bit 4-bit | 0.765 | ±0.000 | 8× |
| fourier D=5K | **0.772** | +0.007 | 8× |

### Mao (290 docs × 30 queries, bge-zh, D=10K)

| 编码 | R@5 | 备注 |
|------|-----|------|
| bipolar D=10K | 1.00 | 饱和（语料库太小） |
| sparse_ternary D=1.3M sp=0.95 | — | 用于单语义发现，非检索 |

---

## 维度 (D) 的相变行为

| D 范围 | 行为 | 适用编码 |
|--------|------|---------|
| 1K-30K | 噪声区。所有信号在随机噪声之下 | 都不适用 |
| 30K-200K | 过渡区。强信号开始出现，弱信号仍在噪声下 | bipolar, simhash |
| 200K-500K | 相变点。单语义信号突然跃升 | sparse_ternary |
| 500K-1.3M | 稳定区。信号随 D 单调增长 | sparse_ternary, simhash |
| 1.3M+ | 未探索。可能饱和 | 待测 |

**关键发现 (论文):** D=200K 是所有编码的相变阈值。低于这个值，HDC 在单语义发现上无效。高于这个值，信号随 D 单调增长。Mao 语料库的 D-scaling 因子为 4.3×（D=200K→1.3M）。

---

## 编码组合可能性（每行一个问题）

| 组合 | 问题 | 状态 |
|------|------|------|
| Fourier + sparse | Fourier 编码后做 top-k 稀疏化？ | 🧪 可能保留傅里叶的精度优势 + 稀疏的存储优势 |
| sparse_ternary × Fourier bind | 稀疏向量做傅里叶 bind？ | 🧪 bind 是角度相加，但稀疏向量有很多 0——零怎么处理？ |
| multi_bit + weighted_bundle | 多级幅度 + 加权——天然匹配 | 🧪 多级幅度已经保留了"重要性"信息 |
| 不同 D 混合 | 查询用 D=10K，文档用 D=100K？ | ❌ 不同 D 的向量不可比较（维度集中定理要求同 D） |
| 编码 × 语料库 | 长文本用 sparse，短文本用 fourier？ | 🧪 长文本在 sparse 下信号更强（论文 发现） |

---

## 选择决策树

```
需要做什么？
│
├── 快速检索，对质量要求不高
│   → bipolar (最低成本，32× 压缩)
│
├── 通用检索，稳定第一
│   → simhash D=10K-100K
│
├── 高精度检索（需要匹配 dense fp32）
│   → multi_bit 4-bit (100% dense 质量，8× 压缩)
│   → 或 fourier D=5K (可能超越 dense)
│
├── 组合推理（需要反复 bind/unbind）
│   → fourier_encode (角度相加，零噪声)
│
├── 单语义发现（找哪些维度在"专心"编码一个概念）
│   → sparse_ternary D≥200K sp=0.95
│
└── 边缘设备部署
    → bipolar + CoreML 1-bit ANE (99.95% fidelity, 31× 压缩)
```

---

## 已确认的编码 × 场景匹配

| 场景 | 编码 | D | 来源 |
|------|------|-----|------|
| RAG 检索 (Mao) | simhash bipolar | 10K-100K | RAG 检索 |
| 单语义发现 | sparse_ternary | 1.3M, sp=0.95 | 论文 |
| 组合推理 (role-binding) | simhash bipolar + bind | 10K | 论文 §4.6 |
| 多值检索 (Qwen) | multi_bit 3-bit | 5K | Path F |
| 芯片边缘部署 | bipolar + CoreML | 10K | Path E |
| 速度极限 | Rust NEON popcount | 10K | 动作库 |
| 跨模型对齐 | simhash + Procrustes | 100K | Steelman C |
| 持续学习 | simhash + bundle 增量 | 30K | hdc_wins_continual |
| 噪声鲁棒 | bipolar | 10K-100K | hdc_wins_noise (100% @ 30% bit-flip) |

---

## 待归档

- [ ] Fourier 在不同 D 下的完整 D-sweep（目前只有 D=5K 和 D=10K）
- [ ] sparse_ternary 在不同 sparsity (0.9, 0.95, 0.99, 0.999) 下的对比
- [ ] Walsh-Hadamard VSA 的检索性能
- [ ] qFHRR (量化傅里叶) vs 标准傅里叶的比较
- [ ] 编码组合的实际测试（Fourier+sparse, multi_bit+weighted）
