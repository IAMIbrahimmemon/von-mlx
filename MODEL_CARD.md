---
library_name: mlx
base_model: wfzyx/von-1.0
pipeline_tag: text-classification
license: apache-2.0
language:
  - en
tags:
  - mlx
  - apple-silicon
  - modernbert
  - text-classification
  - zero-shot-classification
  - nli
  - encoder
  - decision-making
  - quantized
model-index:
  - name: von-1.0-mlx
    results: []
---

# von-1.0-mlx

**Apple Silicon (MLX) port of [`wfzyx/von-1.0`](https://huggingface.co/wfzyx/von-1.0)** — a
non-autoregressive *System One* decision model: a bidirectional **ModernBERT-Large** encoder
fine-tuned for calibrated discrete, probabilistic and ordinal decisions in a single forward
pass.

[![MLX](https://img.shields.io/badge/runtime-MLX-black)](https://github.com/ml-explore/mlx)
[![Apple Silicon](https://img.shields.io/badge/hardware-Apple%20Silicon-lightgrey)](https://huggingface.co/IAMIbrahim/von-1.0-mlx)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://huggingface.co/IAMIbrahim/von-1.0-mlx)
[![Base Model](https://img.shields.io/badge/base-wfzyx%2Fvon--1.0-orange)](https://huggingface.co/wfzyx/von-1.0)

```python
from von_mlx import VonEngine

engine = VonEngine("von-1.0-mlx/8bit")            # quantization auto-detected

r = engine.evaluate(
    state="Database replication lag on cluster us-west-2 exceeded 45 seconds.",
    questions={
        "domain": {
            "type": "choice",
            "instructions": "Classify the root cause domain of this incident.",
            "criteria": {
                "infrastructure": "Database, hardware, network, or server failures",
                "billing":        "Invoices, payments, refunds, subscription queries",
            },
        },
        "blocking": {"type": "noul",
                     "instructions": "Is this actively blocking customer operations?"},
        "severity": {"type": "score",
                     "instructions": "Rate the incident severity.",
                     "criteria": ["Low", "Medium", "High", "Critical"]},
    },
)

r.answers["domain"].choice        # 'infrastructure'
r.answers["domain"].confidence    # 0.992
r.answers["domain"].probabilities # {'infrastructure': 0.9958, 'billing': 0.0042}
r.answers["blocking"].noul        # 0.8093
r.answers["severity"].score       # 2.62   (expected level over Low/Medium/High/Critical)
```

---

## TL;DR

| | |
|---|---|
| **What it is** | MLX port of Von — a calibrated decision model, *not* a generative LLM |
| **Base architecture** | ModernBERT-Large · 28 layers · hidden 1024 · 16 heads · 395.8M params |
| **Two heads** | `option_marker` (98.66% val, **recommended**) · `nli` (96.43% val) |
| **Variants** | `8bit/` **422 MB — recommended** · `fp16/` 791 MB — exact reference |
| **Latency** | ≈ **36 ms** per decision on M3 · 61 ms for a 3-question fan-out |
| **Parity vs PyTorch** | **1.4e-05** max logit Δ (fp32) · **0 argmax flips** on shipped artifacts |
| **Runtime** | MLX only — no PyTorch needed for inference |
| **Output** | JSON answer objects with probabilities — never generated prose |

> **This model does not generate text.** It returns calibrated distributions over options
> you supply. If you need prose, use an LLM and call Von to make the decisions inside it.

---

## Model description

Von answers structured decision problems in a **single bidirectional forward pass** rather
than decoding tokens autoregressively. It supports three decision shapes:

| Type | Question shape | Returns |
|---|---|---|
| `choice` | pick one of `K` labelled options | full probability distribution + confidence margin |
| `noul` | does condition *X* hold? | calibrated probability in `[0, 1]` |
| `score` | rate on an ordered scale of `K` levels | expected value in `[0, K-1]` + per-level distribution |

Because the encoder is bidirectional, every option attends to the state **and to every other
option** at once, so fan-out over `K` options costs **one** forward pass instead of `K`.

### The two heads

Von ships two *independent fine-tunes* — these are not shared encoders. Every tensor differs
(max\|Δ\| ≈ 5e-3 between the two encoders), so both are converted separately.

| Head | Source | Val acc | Temperature | Method |
|---|---|---|---|---|
| **`option_marker`** ← recommended | `option_marker.pt` | **98.66%** | **2.2** | one pass over `K` `[MASK]` markers · 134 task heads |
| `nli` | `model.safetensors` | 96.43% | 1.1692 | one cross-encoder pass per option |

The `option_marker` head packs the state, the question and all `K` candidates into one
sequence:

```text
[CLS] {question} {state} [SEP] [MASK] {option_1} [MASK] {option_2} ... [MASK] {option_K} [SEP]
```

and reads a distribution from the hidden state at each `[MASK]` position. This is why it is
both faster and more accurate than the per-option NLI path.

### Specifications

| | |
|---|---|
| Architecture | `ModernBertForSequenceClassification` |
| Parameters | **395,834,371** (`nli`) · 395,310,081 (`option_marker`) |
| Encoder | 394,781,696 params · 28 layers · hidden 1024 · 16 attention heads |
| Intermediate size | 2624 · activation GELU (exact, erf) |
| Vocab / max positions | 50,368 · 2,048 |
| Attention pattern | alternating full / sliding (`full_attention` every 3rd layer) |
| Sliding window | 128 local attention → **64 inclusive** (129-token window) |
| Pooling | mean · classifier activation GELU |
| Special tokens | `[CLS]` / `[SEP]` / `[MASK]` |

---

## Variants

| Directory | Precision | Size | Recommendation |
|---|---|---|---|
| **`8bit/`** | affine, group size 64 | **422 MB** | ✅ **recommended** — 0 argmax flips, ≤ 0.041 probability shift |
| `fp16/` | float16 | 791 MB | exact reference — use to reproduce the parity numbers |
| *(measured, not shipped)* `4bit` | affine, group size 64 | 224 MB | ❌ **not recommended** — flips a decision |

The choice is **measured, not assumed**. A 2/3/4/5/6/8-bit sweep over six decision cases:

| Variant | Worst probability shift | Argmax flips |
|---|---|---|
| fp16 | 2.0e-03 | 0/6 |
| **8bit** | **4.1e-02** | **0/6** |
| 6bit | 5.8e-02 | 0/6 |
| 5bit | 1.5e-01 | 1/6 |
| 4bit | 3.8e-01 | 1/6 |

4-bit widens the calibrated probability spread by up to **0.38** and flips one of six
decisions. Since calibration is the entire point of this model, **8-bit is the smallest safe
width**. The decision heads (`classifier`, `scorer`, `head`) stay in fp16 in the quantized
variant — they are <1% of parameters and carry the calibration.

---

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx transformers pydantic numpy

# fetch the weights
uv pip install --python .venv/bin/python "huggingface_hub[cli]"
hf download IAMIbrahim/von-1.0-mlx --local-dir von-1.0-mlx
```

```python
from von_mlx import VonEngine

engine = VonEngine("von-1.0-mlx/8bit")     # or fp16/
```

The port code lives at **[github.com/IAMIbrahim/von-mlx](https://github.com/IAMIbrahim/von-mlx)** —
install it from a checkout, or add the checkout root to `PYTHONPATH`.

> **Always run with `PYTHONPATH` unset** (`env -u PYTHONPATH .venv/bin/python …`). Agent
> runtimes and some shells export their own `PYTHONPATH`, which puts a foreign interpreter's
> `site-packages` ahead of the venv's and surfaces as spurious numpy/MLX import failures.

### As an MCP server (recommended for agents)

The cleanest way to put Von behind an agent — Hermes, Claude Desktop, any MCP host:

```json
{
  "mcpServers": {
    "von": {
      "command": "/path/to/von-mlx/von-mlx-mcp"
    }
  }
}
```

Exposes five tools: **`decide`**, **`judge`**, **`rate`**, **`system_one`** (multi-question
fan-out) and **`model_info`**. Model loading is lazy, so the MCP handshake returns in ~0.4 s
and the first tool call pays the ~2.3 s weight-load.

### As an HTTP service

```bash
von-mlx --model-dir von-1.0-mlx/8bit serve --port 8100
```

```bash
curl -X POST http://localhost:8100/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"state":"Disk volume /var/log at 98% capacity.",
       "questions":{"needs_action":{"type":"noul",
         "instructions":"Does this require operational intervention?"}}}'
```

Serves `POST /v1/systemone` (TypeSafe envelope) and `POST /v1/chat/completions` (OpenAI
schema). The OpenAI route is a convenience envelope, **not** an LLM interface — it returns a
JSON answer object and never emits prose.

---

## Verification

Every number below was produced by **executing** the harnesses in the code repo against the
original PyTorch checkpoints. The port is validated against PyTorch, not against itself.

### Gate 1 — port equivalence (MLX fp32 vs PyTorch fp32)

Same weights, two runtimes: this isolates the port from storage precision.

```bash
python verify/parity_check.py --model-dir hf_orig --src hf_orig --gate architecture
```

| Path | max\|Δ\| vs PyTorch | argmax agreement |
|---|---|---|
| NLI head, batch of 4 sentence pairs | **1.36e-05** | 4/4 |
| option-marker, K=3 packed | **1.07e-05** | 3/3 |
| option-marker, K=2 packed | **8.35e-07** | 2/2 |

A **control** ran first: HF float32 vs HF float64 on the same input reaches 4.1e-02 at
layer 27 (hidden magnitudes ~2.2e4). That establishes the fp32 noise floor — the port sits
orders of magnitude below it.

### Gate 2 — decision fidelity of the shipped artifacts

```bash
python verify/parity_check.py --model-dir out/von-1.0-mlx/8bit --src hf_orig --gate artifact
```

Here raw logit deltas are *larger by design* — fp16 storage and 8-bit weights round the
weights — so a raw-logit tolerance is the wrong criterion. This gate requires **zero argmax
flips** and reports deltas for information:

| Artifact | NLI Δ / flips | marker K=3 Δ / flips | marker K=2 Δ / flips |
|---|---|---|---|
| `fp16/` | 4.7e-03 / 0 | 2.6e-03 / 0 | 2.2e-02 / 0 |
| `8bit/` | 4.9e-02 / 0 | 9.3e-02 / 0 | 7.5e-03 / 0 |

> Two gates exist on purpose. Gating a *quantized* model on raw logits either fails a
> perfectly usable artifact or — if you widen the tolerance until it passes — hides a flipped
> decision. Probabilities are what a caller acts on, so decisions are what the gate checks.

### End-to-end against the official SDK

MLX engine vs `von.models.option_marker.OptionMarkerModel` on identical hand-built prompts,
including the zero-shot Noul polarity-cancellation path:

| Case | max logit Δ (MLX vs SDK) | post-temperature max prob Δ |
|---|---|---|
| choice (K=3) | 1.07e-05 | 1.8e-08 |
| choice (K=2) | 8.94e-06 | 6.6e-07 |
| noul, zero-shot debiased | 5.72e-06 | 4.8e-07 |
| noul, explicit criteria | 1.19e-05 | 6.0e-08 |
| score (K=3) | 6.44e-06 | 1.2e-07 |
| score (K=4) | 6.99e-05 | 1.1e-06 |

### Latency

Measured on **M3 (24 GB)**, `option_marker`, batch size 1, including zero-shot debiasing.

| Path | Metric | Time |
|---|---|---|
| MCP tool call, warm (`8bit`) | median | **35.9 ms** |
| MCP 3-question fan-out (`8bit`) | single call | **61.1 ms** |
| HTTP `system_one`, 3 questions (`8bit`) | median | 83.9 ms |
| HTTP `system_one`, 3 questions (`fp16`) | median | 88.7 ms |
| MCP first call | cold (incl. 420 MB load) | 2.35 s |

≈ **24–36 ms per forward pass**. Upstream's "sub-25 ms" figure is the same order but measured
on CUDA; MLX on this M3 is memory-bandwidth bound, so 8-bit quantization recovered only ~5%.

> The upstream `von-sdk` 1.0.1 wheel is missing `von/models/` (packaging bug), so the SDK-side
> reference is loaded from the upstream GitHub source with its SHA-256 pinned in
> `verify/upstream/`.

---

## Implementation notes

Details that are easy to get wrong — each produced *plausible wrong output* rather than a
crash, so they are documented rather than fixed silently:

1. **GELU semantics changed in MLX ≥ 0.31.** `nn.GELU(approx="precise")` now selects the
   **tanh** approximation; exact erf GELU is `approx="none"`. The widely-copied bidirectional
   reference uses `"precise"` and is therefore numerically wrong for HF-compatible encoders
   (max\|Δ\| ≈ 4.7e-4 per evaluation, compounding over 28 layers). Using `"none"` took the
   NLI logit error from 1.2e-2 → 8.1e-06.
2. **RoPE convention.** HF's `apply_rotary_pos_emb` uses split-halves (`rotate_half`), i.e.
   MLX `nn.RoPE(traditional=False)` — the default. `traditional=True` is off by ~6.4. Bases
   differ per attention type: **160000** for `full_attention`, **10000** for
   `sliding_attention`.
3. **Sliding window is 64 inclusive** (`abs(q - kv) <= 64`, a 129-token window) — not
   `local_attention // 2` read as an exclusive half-window.
4. **Layer 0 has no pre-attention LayerNorm** — it is `nn.Identity()`.
5. **Quantize in place**, after loading fp16 weights — never before. A quantized `Linear`
   expects `.scales`/`.biases` that an fp16 state dict does not contain.
6. **Bit width is self-describing.** Each variant stamps `quantization.bits` into its
   `config.json`, and the loader also infers width from packed tensor shapes (a quantized
   weight's last dim becomes `orig * bits / 32`). Building at the wrong width fails to load
   rather than silently misbehaving.
7. **MLX streams are thread-local.** FastAPI's threadpool and `asyncio.to_thread` schedule
   onto threads with no lazily-created CPU stream, so a quantized `nn.Embedding` gather fails
   with `RuntimeError: There is no Stream(cpu, 0) in current thread`. Both servers pin load
   *and* every forward pass to one dedicated thread.

---

## Files

```
8bit/  or  fp16/
├── model.safetensors            # nli head
├── option_marker.safetensors    # option_marker head (recommended)
├── config.json                  # ModernBERT config + quantization stamp
├── tokenizer.json
├── tokenizer_config.json
├── calibration.json             # nli temperature
├── marker_calibration.json      # option_marker temperature
└── von_mlx_manifest.json        # variant manifest
```

Each variant is **self-contained** — no config or tokenizer files are shared between them.

---

## Citation

If you use this port, cite the original model:

```bibtex
@misc{panisa2026von,
  title  = {Von: A Non-Autoregressive System One Decision Model},
  author = {Panisa, Victor},
  year   = {2026},
  url    = {https://huggingface.co/wfzyx/von-1.0}
}
```

## Credits & license

**Apache-2.0**, inherited from the base model.

- Original model, training and SDK — **Victor Panisa**:
  [`wfzyx/von-1.0`](https://huggingface.co/wfzyx/von-1.0) · [`wfzyx/von`](https://github.com/wfzyx/von)
- Base encoder — **ModernBERT**, Answer.AI and LightOn
- MLX encoder written against [Blaizzy/mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) (MIT)

This is a **format port, not a retrain**. Weights are a 1:1 transcription of the original
PyTorch checkpoints; no weights were altered, merged or re-fit.
