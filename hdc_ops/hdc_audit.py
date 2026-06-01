#!/usr/bin/env python3
"""
HDC Audit CLI — 2-Minute Model Fingerprinting Tool
==================================================

A command-line tool for rapid HDC-based model interpretability auditing.
Extracts monosemantic structure from any transformer checkpoint in ~2 minutes.

Usage:
  # Quick audit (2 min): find peak layer and mechanism
  hdc-audit --model Qwen/Qwen2.5-0.5B --corpus corpus.jsonl --output audit.json

  # Deep audit (20 min): full depth trajectory
  hdc-audit --model Qwen/Qwen2.5-0.5B --corpus corpus.jsonl --deep --output audit.json

  # Compare two checkpoints
  hdc-audit --model model_a --corpus corpus.jsonl --output a.json
  hdc-audit --model model_b --corpus corpus.jsonl --output b.json
  hdc-audit --diff a.json b.json

Output Format (JSON):
  {
    "model": "Qwen/Qwen2.5-0.5B",
    "corpus": "corpus.jsonl",
    "audit_type": "quick",
    "peak": {
      "layer": 0,
      "lift": 45.2,
      "mechanism": "sparse",
      "monosem_frac": 0.12,
      "shuf_frac": 0.0013
    },
    "depth_profile": {
      "L0": {"lift": 45.2, "monosem_frac": 0.12},
      "L2": {"lift": 23.1, "monosem_frac": 0.08},
      ...
    },
    "verdict": "embedding-centric model — categorical structure baked into embedding space",
    "recommendations": [
      "Interpretability focus: L0 (embedding layer)",
      "Expected behavior: genre/topic classification should work without fine-tuning",
      "Caution: deep layers (L12+) may have entangled representations"
    ]
  }

Design Principles:
  1. Zero training — load model, run once, get fingerprint.
  2. Minimal dependencies — transformers, numpy, torch.
  3. Sensible defaults — D=200K (phase transition), sp=0.95, mean pooling.
  4. Actionable output — not just numbers, but interpretation and recommendations.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ─── Constants ───────────────────────────────────────────────────────────────
DEFAULT_D = 200_000
DEFAULT_SPARSITY = 0.95
DEFAULT_PURITY = 0.9
DEFAULT_BATCH_SIZE = 8
QUICK_LAYERS = [0, 2, 6, 12]  # L0, early, mid, deep
DEEP_LAYERS = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

# ─── HDC Encoding ────────────────────────────────────────────────────────────
def sparse_ternary_encode(X, D, sp, seed=42):
    N, d_in = X.shape
    k_keep = max(1, int(round((1 - sp) * D)))
    rng = np.random.default_rng(seed)
    chunk_d = 30_000
    chunk_n = min(N, max(8, int(2e8 / D)))
    out = np.zeros((N, D), dtype=np.int8)
    for r0 in range(0, N, chunk_n):
        r1 = min(N, r0 + chunk_n)
        proj = np.zeros((r1 - r0, D), dtype=np.float32)
        for c0 in range(0, D, chunk_d):
            c1 = min(D, c0 + chunk_d)
            W_chunk = rng.standard_normal((d_in, c1 - c0)).astype(np.float32)
            proj[:, c0:c1] = X[r0:r1] @ W_chunk
        magnitudes = np.abs(proj)
        idx = np.argpartition(magnitudes, kth=D - k_keep, axis=1)[:, -k_keep:]
        signs = np.sign(proj[np.arange(r1 - r0)[:, None], idx]).astype(np.int8)
        for ri in range(r1 - r0):
            out[r0 + ri, idx[ri]] = signs[ri]
    return out

# ─── Monosemanticity ─────────────────────────────────────────────────────────
def compute_lift(hv, labels, n_classes, threshold=0.9, chunk=10000):
    N, D = hv.shape
    onehot = np.eye(n_classes, dtype=np.int32)[labels]
    monosem_count = 0
    for c0 in range(0, D, chunk):
        c1 = min(D, c0 + chunk)
        sub = hv[:, c0:c1]
        firing = (sub != 0).astype(np.int32)
        counts = firing.T @ onehot
        total = counts.sum(axis=1)
        maxc = counts.max(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            purity = np.where(total > 0, maxc / np.maximum(total, 1), 0.0)
        monosem_count += int((purity >= threshold).sum())

    rng = np.random.default_rng(0)
    shuf_labels = rng.permutation(labels)
    shuf_onehot = np.eye(n_classes, dtype=np.int32)[shuf_labels]
    shuf_count = 0
    for c0 in range(0, D, chunk):
        c1 = min(D, c0 + chunk)
        sub = hv[:, c0:c1]
        firing = (sub != 0).astype(np.int32)
        counts = firing.T @ shuf_onehot
        total = counts.sum(axis=1)
        maxc = counts.max(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            purity = np.where(total > 0, maxc / np.maximum(total, 1), 0.0)
        shuf_count += int((purity >= threshold).sum())

    real_frac = monosem_count / D
    shuf_frac = shuf_count / D if shuf_count > 0 else 1e-9
    chance = 1 / n_classes
    lift = (real_frac / chance) / (shuf_frac / chance) if shuf_frac > 0 else float('inf')
    return {
        "monosem_count": int(monosem_count),
        "monosem_frac": float(real_frac),
        "shuf_count": int(shuf_count),
        "shuf_frac": float(shuf_frac),
        "lift": float(lift),
    }

# ─── Model & Data Loading ────────────────────────────────────────────────────
def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[hdc-audit] Loading {model_id}...", file=sys.stderr)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    elapsed = time.perf_counter() - t0
    print(f"[hdc-audit]   Loaded in {elapsed:.1f}s", file=sys.stderr)
    return model, tokenizer

def load_corpus(path):
    import json
    chunks = [json.loads(l) for l in open(path)]
    texts = [c["text"] for c in chunks]
    labels_str = [c["layer"] for c in chunks]
    label_names = sorted(set(labels_str))
    l2i = {l: i for i, l in enumerate(label_names)}
    labels = np.array([l2i[s] for s in labels_str], dtype=np.int32)
    return texts, labels, label_names

def extract_activations(model, tokenizer, texts, layers, batch_size=8):
    all_acts = {l: [] for l in layers}
    n = len(texts)
    for b0 in range(0, n, batch_size):
        b1 = min(n, b0 + batch_size)
        batch_texts = texts[b0:b1]
        inputs = tokenizer(batch_texts, return_tensors="pt",
                          padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            for li in layers:
                hs = outputs.hidden_states[li + 1]
                attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
                masked = hs * attention_mask
                summed = masked.sum(dim=1)
                mean_pooled = summed / attention_mask.sum(dim=1).clamp(min=1)
                all_acts[li].append(mean_pooled.float().cpu().numpy())
    return {l: np.concatenate(all_acts[l], axis=0).astype(np.float32) for l in layers}

# ─── Interpretation ──────────────────────────────────────────────────────────
def interpret_fingerprint(peak_layer, peak_lift, peak_mono_frac, n_layers):
    """Generate human-readable interpretation and recommendations."""
    layer_pct = int(peak_layer) / n_layers * 100
    mechanism = "sparse" if peak_mono_frac < 0.25 else "dense"

    if peak_lift < 5:
        verdict = ("WEAK categorical structure. This model may lack robust "
                   "genre/topic abstractions. Suspect for OOD failure.")
        focus = "No clear peak — consider fine-tuning or larger model."
    elif peak_layer == 0:
        verdict = ("EMBEDDING-CENTRIC model. Categorical structure is baked "
                   "into the embedding space. Tokenizer/initialization has "
                   "already performed genre disambiguation.")
        focus = "L0 (embedding layer)"
    elif layer_pct < 15:
        verdict = ("ATTENTION-CENTRIC model. Early attention blocks (L1-L4) "
                   "perform critical context integration to disambiguate "
                   "surface lexical patterns into genre structure.")
        focus = f"L{peak_layer} (early attention)"
    elif layer_pct < 50:
        verdict = ("MIDDLE-PEAK model. Categorical structure emerges at "
                   "intermediate depth, suggesting hierarchical feature "
                   "extraction (lexical → syntactic → semantic).")
        focus = f"L{peak_layer} (mid-depth)"
    else:
        verdict = ("DEEP-PEAK model. Categorical abstraction happens late, "
                   "possibly as a byproduct of next-token prediction rather "
                   "than explicit category learning.")
        focus = f"L{peak_layer} (deep layer)"

    recommendations = [
        f"Interpretability focus: {focus}",
        f"Mechanism: {mechanism} monosemanticity ({peak_mono_frac*100:.1f}% of dims)",
    ]

    if mechanism == "sparse":
        recommendations.append(
            "Sparse mechanism: few dimensions carry strong signal — "
            "efficient but fragile to ablation."
        )
    else:
        recommendations.append(
            "Dense mechanism: many dimensions carry weak signal — "
            "robust to ablation but harder to interpret individually."
        )

    if peak_lift > 50:
        recommendations.append(
            "Exceptional lift (>50×): near-perfect categorical separation. "
            "Ideal for zero-shot classification and content auditing."
        )
    elif peak_lift > 20:
        recommendations.append(
            "Strong lift (20-50×): robust categorical structure. "
            "Suitable for classification and model comparison."
        )
    else:
        recommendations.append(
            "Moderate lift (5-20×): usable categorical structure. "
            "Consider D-scaling or adaptive projection for better signal."
        )

    return verdict, recommendations

# ─── Diff ────────────────────────────────────────────────────────────────────
def diff_audits(path_a, path_b):
    a = json.load(open(path_a))
    b = json.load(open(path_b))

    print("=" * 60)
    print("HDC AUDIT DIFF")
    print("=" * 60)
    print(f"\nModel A: {a['model']}")
    print(f"Model B: {b['model']}")

    pa = a['peak']
    pb = b['peak']

    print(f"\n{'Metric':<20} {'Model A':<15} {'Model B':<15} {'Δ':<10}")
    print("-" * 60)
    print(f"{'Peak layer':<20} {pa['layer']:<15} {pb['layer']:<15} {pb['layer']-pa['layer']:<10}")
    print(f"{'Lift':<20} {pa['lift']:<15.2f} {pb['lift']:<15.2f} {pb['lift']-pa['lift']:<+10.2f}")
    print(f"{'Mechanism':<20} {pa['mechanism']:<15} {pb['mechanism']:<15}")
    print(f"{'Mono %':<20} {pa['monosem_frac']*100:<15.2f} {pb['monosem_frac']*100:<15.2f}")

    # Determine which is "better" for interpretability
    if pa['lift'] > pb['lift'] * 1.5:
        winner = "Model A"
        reason = "significantly higher lift"
    elif pb['lift'] > pa['lift'] * 1.5:
        winner = "Model B"
        reason = "significantly higher lift"
    elif pa['layer'] == 0 and pb['layer'] > 0:
        winner = "Model A"
        reason = "embedding-centric (faster audit, no forward pass needed)"
    elif pb['layer'] == 0 and pa['layer'] > 0:
        winner = "Model B"
        reason = "embedding-centric (faster audit, no forward pass needed)"
    else:
        winner = "Tie"
        reason = "comparable interpretability signal"

    print(f"\n→ {winner} wins for interpretability: {reason}")

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="HDC Model Audit — 2-minute interpretability fingerprinting"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSONL")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--deep", action="store_true", help="Full depth trajectory (slower)")
    parser.add_argument("--D", type=int, default=DEFAULT_D, help="HDC dimensionality")
    parser.add_argument("--sparsity", type=float, default=DEFAULT_SPARSITY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--diff", nargs=2, metavar=("A", "B"), help="Diff two audit JSONs")
    args = parser.parse_args()

    # Diff mode
    if args.diff:
        diff_audits(args.diff[0], args.diff[1])
        return

    # Audit mode
    t0_total = time.perf_counter()

    # Load data
    texts, labels, label_names = load_corpus(args.corpus)
    n_classes = len(label_names)
    print(f"[hdc-audit] Corpus: {len(texts)} chunks, {n_classes} classes", file=sys.stderr)

    # Load model
    model, tokenizer = load_model(args.model)
    n_layers = model.config.num_hidden_layers

    # Determine layers
    layers = DEEP_LAYERS if args.deep else QUICK_LAYERS
    layers = [l for l in layers if l < n_layers]
    if n_layers - 1 not in layers:
        layers.append(n_layers - 1)
    print(f"[hdc-audit] Testing layers: {layers}", file=sys.stderr)

    # Extract activations
    t0 = time.perf_counter()
    acts = extract_activations(model, tokenizer, texts, layers, args.batch_size)
    del model, tokenizer
    import gc
    gc.collect()
    print(f"[hdc-audit] Extraction: {time.perf_counter()-t0:.1f}s", file=sys.stderr)

    # HDC encode + compute monosemanticity per layer
    t0 = time.perf_counter()
    depth_profile = {}
    for li in layers:
        hv = sparse_ternary_encode(acts[li], args.D, args.sparsity)
        mono = compute_lift(hv, labels, n_classes)
        depth_profile[f"L{li}"] = mono
        print(f"[hdc-audit]   L{li}: lift={mono['lift']:.2f}x, "
              f"mono={mono['monosem_frac']*100:.2f}%", file=sys.stderr)

    print(f"[hdc-audit] HDC analysis: {time.perf_counter()-t0:.1f}s", file=sys.stderr)

    # Identify peak
    peak_layer = max(depth_profile.keys(), key=lambda k: depth_profile[k]['lift'])
    peak = depth_profile[peak_layer]
    peak_li = int(peak_layer[1:])

    # Interpretation
    verdict, recommendations = interpret_fingerprint(
        peak_li, peak['lift'], peak['monosem_frac'], n_layers
    )

    # Build output
    output = {
        "model": args.model,
        "corpus": str(args.corpus),
        "audit_type": "deep" if args.deep else "quick",
        "config": {
            "D": args.D,
            "sparsity": args.sparsity,
            "n_classes": n_classes,
            "n_samples": len(texts),
        },
        "peak": {
            "layer": peak_li,
            "lift": peak['lift'],
            "mechanism": "sparse" if peak['monosem_frac'] < 0.25 else "dense",
            "monosem_frac": peak['monosem_frac'],
            "shuf_frac": peak['shuf_frac'],
        },
        "depth_profile": depth_profile,
        "verdict": verdict,
        "recommendations": recommendations,
        "wall_seconds": time.perf_counter() - t0_total,
    }

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[hdc-audit] Audit complete in {output['wall_seconds']:.1f}s", file=sys.stderr)
    print(f"[hdc-audit] Saved: {out_path}", file=sys.stderr)
    print(f"\n[hdc-audit] VERDICT: {verdict}", file=sys.stderr)
    for rec in recommendations:
        print(f"[hdc-audit]   → {rec}", file=sys.stderr)


if __name__ == "__main__":
    main()
