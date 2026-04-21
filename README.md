# Loop Adapter

A finetuning option that grafts an Ouro-style **Looped Language Model** mechanism onto a pretrained Qwen3.5(-VL) checkpoint. Instead of pretraining a LoopLM from scratch, we reuse the existing layer stack `T_max` times per forward pass, attach a small learned **exit gate**, and let LoRA + a tiny inter-loop normalization adapt the backbone to its new recurrent regime.

The original paper: [arXiv:2510.25741](https://arxiv.org/abs/2510.25741) — ByteDance Seed et al. A condensed, implementation-focused summary lives at `documents/LoopLM_Ouro_condensed.md`.

---

## What this adds

| Component | Where |
|-----------|-------|
| Adapter module: zero-init exit gate + learnable inter-loop RMSNorm | `src/model/looplm.py` |
| Loss helpers: `exit_distribution`, `stage1_loss`, `stage2_gate_loss` | `src/model/looplm.py` |
| Looped forward (patches `Qwen3_5TextModel.forward`) | `src/train/monkey_patch_loop.py` |
| Adapter installation + config stamping | `src/model/load_model.py` (`install_loop_adapter`) |
| Training-loop wiring (LoRA exclusion, freeze enforcement, warm-start, optional `torch.compile`) | `src/train/train_sft.py` |
| Custom `compute_loss` (masked, per-step) | `src/trainer/sft_trainer.py` |
| New `TrainingArguments` fields (`loop_*`, `load_from_loop_checkpoint`, `loop_compile_*`) | `src/params.py` |
| Example training scripts | `scripts/finetune_lora_qwen35_loop*.sh` |
| ZeRO-1 config for loop training | `scripts/zero1.json` |

---

## Mental model

A standard decoder-only forward pass is one application of the layer stack:

```
F(x) = lm_head( norm( layer_L( ... layer_1( embed(x) ) ... ) ) )
```

A LoopLM applies the same stack `T` times with shared weights:

```
F^(T)(x) = lm_head( norm( stack( ... stack( embed(x) ) ... ) ) )   # T applications
```

`T = 1` recovers the vanilla model. After each pass `t = 1..T_max`, a small gate produces an instantaneous exit probability `λ_t ∈ (0, 1)` per position. The exit-step distribution `p_φ(t | x)` is built from the survival product of the `λ`s.

Training objective (Stage 1):

```
L = Σ_t p_φ(t | x) · L^(t)   −   β · H( p_φ(·|x) )
```

The first term is the expected per-step cross-entropy under the gate's current preferences. The second is an entropy bonus that prevents the gate from collapsing (uniform-prior ELBO view).

Stage 2 freezes the LM and trains only the gate against a supervised "should I keep looping?" label built from realized loss improvement between successive steps.

---

## Quick start

### 1. Convert the dataset

```bash
python convert_sonnet.py
# → train.json
```

(There's also `convert.py` for the smaller opus dataset, and `convert_sonnet.py` accepts `--limit N`, `--difficulty`, `--category` for filtering.)

### 2. Train (single A100, 80GB, T_max=2)

```bash
bash scripts/finetune_lora_qwen35_loop_t2.sh
```

This is the recommended starting point. Runs Stage 1 (joint LM-via-LoRA + gate training) for one epoch over 122K examples at `T_max=2`. Sized for a single A100 80GB, no DeepSpeed.

For T=4 or multi-GPU setups:

```bash
bash scripts/finetune_lora_qwen35_loop.sh             # T=4 default
bash scripts/finetune_lora_qwen35_loop_stage2.sh      # Stage 2 (gate-only) follow-up
```

### 3. Read the loop diagnostics

The trainer logs these every step (alongside standard `loss`, `learning_rate`, etc.):

| Metric | Meaning |
|--------|---------|
| `loop/ce_t1`, `loop/ce_t2`, ... | Mean CE if the model exited at step `t`. Watch whether `ce_t≥2` drops below `ce_t1` over training. |
| `loop/expected_ce` | `Σ p_φ(t) · L^(t)` — the loss term being minimized. |
| `loop/entropy` | Shannon entropy of `p_φ` (max = `ln(T_max)`). Should slowly decrease but not collapse. |
| `loop/mean_exit_step` | `E[t]` — should drift away from `T_max/2` as the gate forms preferences. |

What to look for: see the **Diagnostics** section below.

---

## CLI flags

All exposed via `TrainingArguments`. Pass on the command line as `--loop_<name> <value>`.

| Flag | Default | Notes |
|------|---------|-------|
| `loop_enable` | `False` | Master switch. When false, training is a normal LoRA SFT run. |
| `loop_t_max` | `4` | Number of recurrent passes per forward. Start with `2` if unsure. |
| `loop_stage` | `1` | `1` = joint LM+gate. `2` = freeze LM, train gate only. |
| `loop_beta` | `0.1` | Entropy regularizer weight. Lower (`0.05`) gives the gate more freedom. |
| `loop_inter_norm` | `True` | Insert learnable RMSNorm between loop iterations. Mitigates residual blowup on Qwen3.5 (which is pre-norm, not sandwich-norm). |
| `loop_gate_only` | `False` | Stage 2 helper: freeze LoRA too, train only the gate. |
| `loop_share_lm_head` | `True` | Reuse base `lm_head` for per-step readouts. |
| `loop_kv_cache_strategy` | `"last"` | Inference-only: `full` \| `last` \| `avg`. Stamped onto config. |
| `loop_stage2_adaptive_k` | `50.0` | Sharpness `k` for the stage-2 BCE label. |
| `loop_stage2_adaptive_gamma` | `0.005` | Improvement threshold `γ`. |
| `load_from_loop_checkpoint` | `None` | Warm-start: dir containing a Stage-1 LoRA + `non_lora_state_dict.bin`. |
| `loop_compile_layers` | `False` | Apply `torch.compile` to each decoder layer (per-layer is the safe granularity for shared-weight loops). |
| `loop_compile_mode` | `"default"` | `default` \| `reduce-overhead` \| `max-autotune`. |

---

## Stage 1 → Stage 2

Stage 1 trains LoRA + the loop adapter jointly. Stage 2 freezes the LM (and LoRA) and refines only the gate against the adaptive BCE label, which simultaneously penalizes underthinking (exiting when the next loop would still help) and overthinking (continuing when gains have stalled).

```bash
# Stage 1
bash scripts/finetune_lora_qwen35_loop.sh

# Stage 2 — warm-starts from Stage-1 output dir
STAGE1_DIR=output/qwen35_lora_loop_stage1 bash scripts/finetune_lora_qwen35_loop_stage2.sh
```

Per the paper, Stage 2 buys ~2–3% on MMLU at matched compute. Skip it for a first pass.

---

## Diagnostics: how to tell it's working

Three independent questions, three independent signals. Don't conflate them.

### Is the second loop helping at all?

Watch `ce_t2 - ce_t1`. Healthy trajectory:

| Step range | What you want |
|------------|---------------|
| 0–100      | `ce_t2 > ce_t1` (backbone hasn't been adapted yet) |
| 100–500    | Gap shrinks toward 0.1 |
| 500–2000   | `ce_t2 ≤ ce_t1` on most batches |
| 2000+      | `ce_t2` stably 0.02–0.1 *below* `ce_t1` |

If the gap doesn't shrink by ~step 1000, the loop isn't paying off on your data. The gate will (correctly) collapse to always-exit-at-`t=1`.

### Is the gate learning input-conditional preferences?

Watch `loop/mean_exit_step` and `loop/entropy`:

- `mean_exit_step` drifting away from `T_max/2` and varying across batches → gate is forming preferences.
- `entropy` slowly decreasing toward 0.3–0.5 (without crashing to 0) → gate is committing without losing exploration.
- Both pinned at their initial values → gate isn't learning at all.

### Is the looped model actually better than a plain LoRA?

You need a baseline. Run `finetune_lora_qwen35.sh` with the same data, same LR, same step count, and compare on a held-out eval set. The training loss is biased by the loop's `expected_ce` formulation and is not directly comparable.

### Red flags

- `NaN` / `Inf` in any `loop/*` → numerical instability. Drop `T_max` to 2 or disable `loop_inter_norm`.
- `ce_t2 > 5` persistently → second loop is destroying the representation. Try `--loop_inter_norm False`.
- Loss flat for >500 steps → LoRA isn't getting gradient. Check LR and warmup.

---

## Performance

Speed wins, in priority order:

1. **Install the fast linear-attention path.** Qwen3.5 has hybrid layers (3 linear-attention : 1 full-attention by default). Without `flash-linear-attention` and `causal-conv1d` installed, the linear-attention layers fall back to a slow PyTorch reference and dominate step time.
   ```bash
   pip install flash-linear-attention
   pip install causal-conv1d         # may need a CUDA-toolkit-matching torch build
   pip install flash-attn --no-build-isolation
   ```
2. **Enable Flash Attention 2** for the full-attention layers: drop `--disable_flash_attn2 True`.
3. **Disable gradient checkpointing** if memory permits. At `T=2` on 80GB you usually have room.
4. **Try `--loop_compile_layers True`** (per-layer `torch.compile`). First ~20 steps are slow; subsequent steps should be faster.

Memory napkin (Qwen3.5-4B, LoRA r=32, bf16, grad-ckpt on, T=2, B=2, S=4096): ~27 GB total. Bump batch or sequence to use more.

---

## DeepSpeed compatibility

**Don't use DeepSpeed ZeRO-1/2 with looping on a single GPU.** ZeRO installs a per-param grad-reduce hook that fires once per backward edge; the loop reuses each LoRA param T times per forward, so the hook trips an "already reduced" assertion. The provided scripts launch with plain `python` for this reason.

For multi-GPU, ZeRO-3 may work (different hook semantics) but is untested with this code. If you need it, start there as the first thing to verify on your setup.

---

## Inference

**Not yet implemented.** The patched forward raises `NotImplementedError` when `use_cache=True`. Cached generation needs a per-loop cache wrapper (the paper's "last-step-only KV during decode" trick saves 4× memory but isn't wired up here).

Training, eval-loss, and held-out evaluation against a non-cached forward all work today.

---

## Known risks specific to Qwen3.5

The Ouro paper studied looping on a vanilla decoder transformer with sandwich-norm. Qwen3.5 differs in two material ways:

1. **Hybrid attention** — ~3/4 of the layers are gated DeltaNet (linear attention with recurrent state). Looping these is unstudied; their inductive bias under reuse is materially different from self-attention.
2. **Pre-norm, not sandwich-norm** — residual stream accumulates `T × num_layers` worth of sublayer outputs without external normalization. The `loop_inter_norm` flag (on by default) inserts a learnable RMSNorm between iterations as mitigation, but it is not a guarantee.

If T=4 is unstable, drop to T=2. If T=2 is unstable, the conclusion is that this backbone doesn't loop cleanly and you should either pick a different starting checkpoint (e.g., a pure full-attention Qwen3 variant) or accept that looping doesn't help on this stack.

---

## Implementation map

```
src/
├── model/
│   ├── looplm.py                # LoopAdapter + loss helpers
│   └── load_model.py            # install_loop_adapter
├── train/
│   ├── monkey_patch_loop.py     # looped Qwen3_5TextModel.forward
│   └── train_sft.py             # CLI wiring, freezing, warm-start, compile
├── trainer/
│   └── sft_trainer.py           # compute_loss override
└── params.py                    # loop_* TrainingArguments fields

scripts/
├── finetune_lora_qwen35_loop_t2.sh         # T=2, single A100
├── finetune_lora_qwen35_loop.sh            # T=4 default
├── finetune_lora_qwen35_loop_stage2.sh     # Stage 2
└── zero1.json                              # ZeRO-1 config (for multi-GPU)

documents/
├── LoopLM_Ouro_condensed.md                # Paper summary
├── looplm_finetune_plan.md                 # Implementation plan
└── looplm_finetune_plan_feedback.md        # Plan review
```
