# von-mlx

Apple Silicon (**MLX**) port of [`wfzyx/von-1.0`](https://huggingface.co/wfzyx/von-1.0) —
a non-autoregressive *System One* decision model built on **ModernBERT-Large** (395M
params, 28 layers). It answers structured decision problems (pick one of K, yes/no
probability, ordinal score) in a single forward pass instead of generating tokens.

Published MLX weights: **https://huggingface.co/IAMIbrahim/von-1.0-mlx**

This repository is the **port code** — the MLX implementation, the PyTorch→MLX converter,
and the verification harnesses. It is a format port, not a retrain: weights are a 1:1
transcription of the original checkpoints.

```python
from von_mlx import VonEngine

engine = VonEngine("von-1.0-mlx/8bit")   # quantization auto-detected
r = engine.evaluate(
    state="Database replication lag on cluster us-west-2 exceeded 45 seconds.",
    questions={
        "domain": {"type": "choice",
                   "instructions": "Classify the root cause domain of this incident.",
                   "criteria": {"infrastructure": "Database, hardware, network failures",
                                "billing": "Invoices, payments, refunds"}},
        "blocking": {"type": "noul",
                     "instructions": "Is this actively blocking customer operations?"},
        "severity": {"type": "score",
                     "instructions": "Rate the incident severity.",
                     "criteria": ["Low", "Medium", "High", "Critical"]},
    },
)
print(r.answers["domain"].choice, r.answers["domain"].confidence)   # infrastructure 0.98
print(r.answers["blocking"].noul)                                   # 0.88
print(r.answers["severity"].score)                                  # 2.4
```

Answers are pydantic models matching the official SDK's field names, so this is a drop-in
swap for `von`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx transformers pydantic numpy fastapi
uv pip install --python .venv/bin/python -r requirements-serve.txt   # uvicorn, httpx
```

`torch` and `safetensors` are needed **only** to convert weights and run the verification
harnesses; inference is pure MLX.

> Always invoke the venv's python as `env -u PYTHONPATH .venv/bin/python`. A Hermes/agent
> session exports its own `PYTHONPATH`, which leaks a foreign Python's `site-packages`
> into the venv and reads as a broken numpy/torch install.

## Convert the weights

```bash
hf download wfzyx/von-1.0 --local-dir hf_orig
env -u PYTHONPATH .venv/bin/python convert.py --src hf_orig --out out/von-1.0-mlx
```

Produces self-contained `fp16/` and `8bit/` variants.

## Use

```bash
# one decision problem from a JSON file or stdin
von-mlx --model-dir out/von-1.0-mlx/8bit decide --problem problem.json

# latency benchmark
von-mlx --model-dir out/von-1.0-mlx/8bit bench

# HTTP server (TypeSafe /v1/systemone + OpenAI /v1/chat/completions)
von-mlx --model-dir out/von-1.0-mlx/8bit serve --port 8100
```

The server is **not an LLM**: `/v1/chat/completions` returns a JSON answer object, never
generated prose. The authoritative endpoint is `/v1/systemone`.

## Verification

The port is checked against PyTorch rather than against itself.

```bash
P="env -u PYTHONPATH .venv/bin/python"

# 1a. PORT equivalence: MLX vs torch, both on the original fp32 weights
$P verify/parity_check.py --model-dir hf_orig --src hf_orig --gate architecture
#   -> nli 1.36e-05 | option-marker 1.07e-05 / 8.35e-07   argmax flips 0

# 1b. SHIPPED ARTIFACT fidelity: gated on 0 argmax flips, not raw logits
$P verify/parity_check.py --model-dir out/von-1.0-mlx/fp16 --src hf_orig --gate artifact
$P verify/parity_check.py --model-dir out/von-1.0-mlx/8bit --src hf_orig --gate artifact

# 2. per-layer divergence localisation + the fp32 noise-floor control
$P verify/layerwise_diff.py --model-dir hf_orig

# 3. end-to-end vs the official von SDK, incl. zero-shot Noul debiasing
$P verify/sdk_parity.py --model-dir hf_orig
#   -> all six cases <= 6.99e-05 on logits

# 4. what quantization costs, at the probability level
$P verify/quant_report.py --src hf_orig \
     --fp16 out/von-1.0-mlx/fp16 --4bit out/von-1.0-mlx/8bit

# 5. bit-width sweep (this is how 8-bit was chosen over 4-bit)
$P verify/bits_sweep.py --src hf_orig
```

### Two gates, on purpose

A raw-logit tolerance is the right test for the **port** (same weights, two runtimes) and
the wrong test for a **quantized artifact** (different weights by construction). Gating a
quantized model on raw logits either fails a perfectly usable artifact or — if you loosen
the number until it passes — hides a flipped decision. So `--gate artifact` requires
**zero argmax flips** and reports logit deltas for information only.

## Why 8-bit and not 4-bit

A 2/3/4/5/6/8-bit sweep over six decision cases:

| variant | worst probability shift | argmax flips |
|---|---|---|
| fp16 | 2.0e-03 | 0/6 |
| **8bit** | **4.1e-02** | **0/6** |
| 6bit | 5.8e-02 | 0/6 |
| 5bit | 1.5e-01 | 1/6 |
| 4bit | 3.8e-01 | 1/6 |

4-bit flips a decision and widens the calibrated probability spread by 0.38. Calibration
is the entire point of this model, so 8-bit is the smallest safe width.

## Two heads, both ported

| head | source | temperature | val acc | method |
|---|---|---|---|---|
| `option_marker` (recommended) | `option_marker.pt` | 2.2 | 98.66% | one pass, K `[MASK]` markers |
| `nli` | `model.safetensors` | 1.1692 | 96.43% | one pass per option |

The two checkpoints are **different fine-tunes**, not shared encoders. Comparing the MLX
option-marker head against the torch NLI encoder produces ~7.6 logit deltas that look like
a port bug but are a harness bug — `verify/parity_check.py` loads each head's own encoder.

## Layout

```
von_mlx/
  config.py     ModelArgs (HF ModernBERT config)
  model.py      MLX encoder + both heads
  load.py       loading, quantization detection, strict key checks
  decision.py   VonEngine -- port of the SDK's OptionMarkerBackend
  types.py      /v1/systemone wire types
  server.py     FastAPI server (systemone + OpenAI shim)
  cli.py        serve / decide / bench / verify
convert.py      PyTorch -> MLX converter (fp16 + 8bit)
verify/         parity, layerwise, SDK-parity, quant report, bit-width sweep
```

## Gotchas worth knowing before editing

Each of these cost real time and produced *plausible* wrong output rather than a crash:

- **`nn.GELU(approx=...)` flipped meaning in MLX ≥ 0.31.** `"precise"` is now the *tanh*
  approximation; exact erf GELU is `"none"`. The widely-copied bidirectional reference
  uses `"precise"` and is therefore numerically wrong for HF-compatible encoders
  (max|Δ| ≈ 4.7e-4 per evaluation, compounding across 28 layers). Fixing this took the
  NLI logit error from 1.2e-2 to 8.1e-06.
- **RoPE convention.** HF ModernBERT uses split-halves (`rotate_half`), which is MLX's
  *default* `nn.RoPE(traditional=False)`. `traditional=True` is off by ~6.4. Bases differ
  per attention type: 160000 for `full_attention`, 10000 for `sliding_attention`.
- **Sliding window is 64** (inclusive, a 129-token window), not `local_attention // 2`
  read as 32.
- **Layer 0 has no `attn_norm`** — it is `nn.Identity()`.
- **Quantize in place**, after loading fp16 weights, never before: a quantized `Linear`
  expects `.scales`/`.biases` that an fp16 dict does not contain.
- **`mx.save_safetensors` needs a flat dict**, not a nested parameter tree — otherwise
  `RuntimeError: std::bad_cast`.
- **MLX streams are thread-local.** FastAPI runs sync endpoints on a threadpool thread
  that has no lazily-created CPU stream, so a quantized `nn.Embedding` gather fails with
  `RuntimeError: There is no Stream(cpu, 0) in current thread`. `server.py` pins load and
  every forward pass to one dedicated thread.

## Credits & license

Apache-2.0 (inherited from the base model).

- Original model, training and SDK: **Victor Panisa** —
  [`wfzyx/von-1.0`](https://huggingface.co/wfzyx/von-1.0), [`wfzyx/von`](https://github.com/wfzyx/von).
- Base encoder: **ModernBERT** — Answer.AI and LightOn.
- MLX encoder written against [Blaizzy/mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) (MIT).

Note: the PyPI `von-sdk` 1.0.1 wheel omits `von/models/` (upstream packaging bug), so
`verify/sdk_parity.py` loads the reference model from the upstream GitHub source, with its
sha256 pinned in `verify/upstream/`.
