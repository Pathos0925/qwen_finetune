# Plan: LoopLM (Ouro-style) finetuning option for Qwen-VL-Series-Finetune

## Goal

Add a `--loop_enable` mode to `finetune_lora_qwen35.sh` that takes a pretrained Qwen3.5-VL checkpoint and finetunes it (via LoRA + a small new gate head) into a Looped Language Model: the LLM layer stack is reapplied `T_max` times per forward pass, with a learned exit gate selecting depth adaptively.

We are not pretraining a LoopLM from scratch. We are *grafting* the loop architecture onto an existing model — the LoRA helps the layers re-purpose for recurrent use, and the new exit-gate head + per-step readouts get learned end-to-end.

## What changes, file by file

### 1. `src/params.py` — new training arguments
Add to `TrainingArguments` (and mirror onto `CLSArguments`/`DPOArguments` only if you want loop in those flows; SFT-first is fine):

- `loop_enable: bool = False` — master switch.
- `loop_t_max: int = 4` — recurrent depth cap (paper uses 4; 8 was unstable).
- `loop_beta: float = 0.1` — entropy regularization weight (drop to 0.05 mid-run).
- `loop_stage: int = 1` — `1` = joint LM+gate (entropy-regularized expected CE); `2` = freeze LM, train only gate with BCE adaptive label.
- `loop_gate_only: bool = False` — Stage 2 helper: freeze everything but the gate.
- `loop_apply_to: str = "language_model"` — clarity flag; loop wraps the LLM stack only, not vision tower or merger.
- `loop_share_lm_head: bool = True` — reuse base `lm_head` for per-step readouts (recommended; keeps the new params tiny).
- `loop_per_step_loss_weighting: str = "exit_dist"` — `exit_dist` (paper) or `uniform` for an ablation knob.
- `loop_kv_cache_strategy: str = "last"` — inference-only: `full` | `last` | `avg`. Stored in config for the patched generate.

### 2. New file `src/model/looplm.py` — the loop adapter

A small module that holds the new trainable head:

```python
class LoopGate(nn.Module):
    def __init__(self, d_model):
        self.gate = nn.Linear(d_model, 1)  # σ → λ_t per position
    def forward(self, h):  # h: [B, S, D] post-final-norm hidden
        return torch.sigmoid(self.gate(h)).squeeze(-1)  # [B, S]
```

Plus pure-function helpers:

- `exit_distribution(lambdas) -> p[T,B,S]` — same recursion as in §8 of the condensed doc.
- `stage1_loss(per_step_logits, labels, lambdas, beta, ignore_index)` — expected token-CE under `p_φ(t|x)` minus `β·H(p_φ)`. Mask out IGNORE_INDEX positions before summing.
- `stage2_gate_loss(per_step_logits, labels, lambdas, k=50.0, gamma=0.005)` — detached per-step CE, build label `w = σ(k·(ΔL − γ))`, BCE between `1 − λ_t` and `w`.

These are framework-free Torch and easy to unit test.

### 3. New file `src/train/monkey_patch_loop.py` — wrap the LLM stack

Replace `Qwen3_5TextModel.forward` (and the MoE variant if/when needed) with a looped variant that:

1. Runs the existing setup (embeddings, position ids, mask construction, rotary embeddings) **once**.
2. Loops the existing `for layer in self.layers: ...` block `T = self.config.loop_t_max` times, **reusing the same parameters each pass**. Same `position_embeddings`, same masks.
3. After each pass:
   - Apply `self.norm(hidden_states)` to get `h^(t)`.
   - Stash `h^(t)` for the loss step (don't compute logits here — defer to the wrapper to keep memory down; or compute logits incrementally if memory allows).
   - Compute `λ_t = self._loop_gate(h^(t))` and stash it.
4. Return a custom `Qwen3_5ModelOutputWithPast`-shaped object whose `last_hidden_state` is `h^(T_max)` (so the existing `lm_head` path still works), but with two extra fields: `per_step_hidden_states: List[Tensor]` and `per_step_lambdas: List[Tensor]`.

Caveats to nail down:
- **KV cache**: during training (`use_cache=False`) this is straightforward — every loop iteration recomputes attention against the same K/V projected from the current hidden state. During inference we'll need to be careful — Qwen3.5's `Qwen3_5DynamicCache` will accumulate per-layer K/V on every loop pass, blowing up memory. Solution: pass a cache wrapper that *resets the layer offsets* between loop iterations during prefill, and during decode reuse only the last-step cache (per the paper's 4× memory-reduction trick). This is the trickiest piece — defer until Stage 1 trains end-to-end.
- **Gradient checkpointing**: each loop iteration must be its own checkpoint segment so we don't keep T copies of activations. Wrap each pass through `self.layers` in `torch.utils.checkpoint.checkpoint`.
- **Position ids / masks** are computed once and reused — they don't depend on the iteration.

### 4. `src/train/monkey_patch_forward.py` — return per-step state

Modify `_qwen3_5_mixed_modality_forward_impl` to forward through `self.language_model(...)` as today, but propagate `per_step_hidden_states` / `per_step_lambdas` from the inner output up to the outer `Qwen3_5ModelOutputWithPast`. Add those fields to a small subclass (don't mutate the upstream dataclass).

### 5. `src/model/load_model.py` — install the gate + loop patch

After `AutoModelForImageTextToText.from_pretrained(...)`, when `training_args.loop_enable`:

1. `apply_qwen_vl_loop_patches(config.model_type)` — installs the looped `forward` (call this *after* the existing patcher).
2. Attach a `LoopGate(d_model)` instance as `model.model.language_model._loop_gate` (so it's reachable from the patched forward) and register it in the module tree so `state_dict` saves it.
3. Stamp `loop_t_max`, `loop_kv_cache_strategy`, etc. onto `model.config` for inference reuse.

The gate is initialized so that `λ ≈ 0.5` initially (Linear with small init + zero bias → σ ≈ 0.5). That keeps early training near-uniform over depth so the LM has time to adapt before the gate sharpens.

### 6. `src/trainer/sft_trainer.py` — custom loss path

Override `compute_loss`:

- If `not args.loop_enable`: call `super().compute_loss(...)` (no behavior change).
- Else: run the model forward, get `per_step_hidden_states` and `per_step_lambdas` from the outputs, project each `h^(t)` through `model.lm_head` (under `torch.no_grad()` for the early steps if `loop_share_lm_head` is set and we want to avoid extra memory… but to actually get gradients through the LM at intermediate depths we need *with* grad — accept the memory cost, or use gradient checkpointing on the lm_head as well).
- Compute `stage1_loss` or `stage2_gate_loss` from §2.
- Log scalar diagnostics: `expected_ce`, `entropy`, `mean_exit_step` (= `Σ t · p_φ(t)`), per-step CE `L^(1)..L^(T)` so you can watch whether deeper steps actually reduce loss.

### 7. `src/train/train_sft.py` — wire it up

- After the existing LoRA setup, if `training_args.loop_enable`:
  - Force `freeze_llm=True` (the LM is frozen except via LoRA — same constraint as today).
  - Always make the gate trainable: walk `model.named_parameters()`, set `requires_grad=True` for any name containing `_loop_gate`.
  - If `training_args.loop_stage == 2` and `training_args.loop_gate_only`: zero out `requires_grad` for everything *except* `_loop_gate` (LoRA included).
- Add validation: refuse `--use_liger_kernel True` with `--loop_enable True` (Liger replaces the LM head's CE; doesn't compose with our per-step readouts).

### 8. Save / load

- `train_utils.get_peft_state_non_lora_maybe_zero_3` already captures non-LoRA trainables → the gate weights will land in `non_lora_state_dict.bin` automatically. Verify after first run.
- For inference, write a small loader that re-applies the loop monkey patch, instantiates `LoopGate`, loads `non_lora_state_dict.bin`, then merges the LoRA. Probably belongs in `src/merge_lora_weights.py` as an `--loop` flag.

### 9. New `scripts/finetune_lora_qwen35_loop.sh`

Clone the existing script and add:

```bash
--loop_enable True \
--loop_t_max 4 \
--loop_beta 0.1 \
--loop_stage 1 \
--learning_rate 5e-5 \   # ~half of the non-loop LR per the paper
--gradient_checkpointing True \
```

Stage-2 follow-up script: same but `--loop_stage 2 --loop_gate_only True --learning_rate 1e-4`, resuming from Stage 1's output dir.

## Order of work

1. **Skeleton + smoke test** (no training): add params, write `LoopGate`, write the looped `forward` patch, confirm one full forward pass produces `T_max` hidden states with sane shapes on a single sample. This shakes out the cache / mask / position-id reuse questions.
2. **Stage 1 loss path**: wire `compute_loss`, run a 50-step overfit on a tiny `train.json` with `T_max=2` to verify gradients flow into both the LoRA adapters and the gate, and that loss decreases.
3. **Scale up to T_max=4**, gradient checkpointing each loop iteration, full training run on a real dataset.
4. **Stage 2 gate refinement**: freeze LM/LoRA, BCE adaptive gate loss, short run.
5. **Inference / KV-cache work**: only after Stages 1–2 are demonstrably training. This is its own chunk and gates the deployable artifact, not the research signal.

## Open questions to resolve before coding

- **Norm placement**: Qwen3.5's blocks are pre-norm, not Ouro's "sandwich" norm. The paper calls sandwich-norm a stability win for deep unrolled graphs. We won't restructure pretrained blocks. Mitigation: smaller `T_max` (≤4), conservative LR, watch for loss spikes — fall back to `T_max=2` if needed.
- **Per-step readout cost**: T_max=4 readouts × full vocab is meaningful memory at long sequence lengths. Options: only score loss tokens (mask first), checkpoint the head, or readout only at `t ∈ {1, T_max}` early in training.
- **Multimodal interaction**: vision features are injected once before the loop. That's the right thing — we're looping the LM, not the vision tower. Just confirm the patched forward only touches `language_model.forward`, leaving the vision/merger path intact.
- **3D RoPE position ids**: Qwen3.5's `forward` reshapes `position_ids` and consumes them in attention. Make sure they're computed once and reused identically across loop iterations (they don't depend on hidden state).

## What this does *not* try to reproduce

- The 7.7T-token pretraining recipe in §4 of the doc — irrelevant; we're finetuning.
- Layer-duplication "upcycling" (1.4B→2.6B) — not applicable to a frozen pretrained Qwen3.5.
- RL stages — paper itself reports they didn't beat SFT.
- Speculative decoding via intermediate readouts (§9) — nice-to-have, deferrable.
