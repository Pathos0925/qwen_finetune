# Feedback on `looplm_finetune_plan.md`

Reviewed against the actual `transformers==5.3.0` `qwen3_5` modeling code at `transformers/models/qwen3_5/modeling_qwen3_5.py`, the existing monkey-patching in `src/train/monkey_patch_forward.py`, the load path in `src/model/load_model.py`, the LoRA wiring in `src/train/train_sft.py`, and the trainer in `src/trainer/sft_trainer.py`.

The plan is well-organized and the loss/gate math matches the Ouro paper. The biggest gaps are about Qwen3.5 itself: it is not a vanilla pre-norm transformer, and the hooking points/shapes in the plan don't quite match the real module hierarchy. Issues are listed roughly in order of severity.

---

## 1. Critical — Qwen3.5 is a hybrid model with stateful linear-attention layers

`Qwen3_5DecoderLayer` (modeling_qwen3_5.py:828) chooses between **two completely different sublayers** based on `config.layer_types[layer_idx]`:

- `"full_attention"` → `Qwen3_5Attention` (standard self-attention with KV cache)
- `"linear_attention"` → `Qwen3_5GatedDeltaNet` (RWKV-style gated DeltaNet with `conv_states` + `recurrent_states`)

The default pattern is `interval_pattern=4` → 3 linear-attention layers for every 1 full-attention layer (configuration_qwen3_5.py:184). For a 4B Qwen3.5 with ~36 layers, that is ~27 stateful SSM layers in the loop.

This breaks two assumptions in the plan:

1. **"During training (`use_cache=False`) this is straightforward — every loop iteration recomputes attention against the same K/V projected from the current hidden state."** Not true for the gated DeltaNet path. Look at `Qwen3_5GatedDeltaNet.forward` (modeling_qwen3_5.py:512): even with `cache_params=None`, each call runs `chunk_gated_delta_rule(...)` to *compress* the sequence into a recurrent state. Re-feeding the layer's output back through itself T times produces a stack of fixed-point compressions, not the recurrent reasoning the Ouro paper describes for self-attention. There's no obvious reason this is unsafe, but the inductive bias is materially different from what the paper studied.
2. **KV-cache during inference** is much harder than the plan suggests. With T loops, you'd need T copies of `key_cache`/`value_cache` (full attention) AND T copies of `conv_states`/`recurrent_states` (linear attention). The "last-step cache only" trick (paper §6) is unstudied for hybrid SSM/attention layers — there is no reason to expect the recurrent state from loop t=4's prefill to be a good initial state for decoding when training was done with no cache at all.

**Recommendation:** add an open question explicitly listing this as a research risk; consider whether to start with `Qwen3-VL` (pure full-attention Qwen3 LLM) instead of `Qwen3.5` as a less risky first target. At minimum, the plan should state which Qwen3.5 checkpoint is being grafted (1.5B / 4B / 30B-A3B) and verify its `layer_types` before any code is written.

---

## 2. Critical — the patch target and the LM head are on different modules

The plan says (§3): *"Replace `Qwen3_5TextModel.forward` (and the MoE variant if/when needed) with a looped variant…"* and (§5): *"Attach a `LoopGate(d_model)` instance as `model.model.language_model._loop_gate`."* That part is right.

But the per-step readouts go through `lm_head`, and **`lm_head` lives at `model.lm_head` on the outer `Qwen3_5ForConditionalGeneration` wrapper** (modeling_qwen3_5.py:1917) — not on `Qwen3_5TextModel` and not on `Qwen3_5Model`. The chain is:

```
Qwen3_5ForConditionalGeneration   (has .lm_head, .model)
    └── Qwen3_5Model              (has .visual, .language_model — this is what monkey_patch_forward.py patches today)
        └── Qwen3_5TextModel       (has .embed_tokens, .layers, .norm — the LLM stack)
```

Consequences the plan misses:

- The patched `Qwen3_5TextModel.forward` cannot call `lm_head` directly — so per-step logits must be computed *outside* the patched function. The plan does this in `compute_loss`, which is fine, but it requires `per_step_hidden_states` to flow through **two more wrappers** to reach the trainer:
  1. `Qwen3_5TextModel.forward` returns a `BaseModelOutputWithPast`.
  2. `Qwen3_5Model.forward` (already monkey-patched in `monkey_patch_forward.py:159`) constructs `Qwen3_5ModelOutputWithPast(**outputs, rope_deltas=...)`. With unknown extra fields, `**outputs` will throw or silently drop them.
  3. `Qwen3_5ForConditionalGeneration.forward` (modeling_qwen3_5.py:1960) does `hidden_states = outputs[0]` and computes only `self.lm_head(hidden_states[:, slice_indices, :])` and a single `loss_function(...)` call. Per-step state is discarded here.

  So the plan must monkey-patch **three** forwards (or pass the per-step list through a side channel like an attribute on `self` or via `kwargs`), not one. §4 only mentions adding fields to `Qwen3_5ModelOutputWithPast`, which silently doesn't help because `Qwen3_5ForConditionalGeneration.forward` ignores them.

- `Qwen3_5TextModel.forward` has decorators `@merge_with_config_defaults`, `@capture_outputs`, `@auto_docstring` (modeling_qwen3_5.py:1315–1317). Direct `.forward = new_fn` assignment skips them. That's actually the existing pattern (see `qwen3_5_mixed_modality_forward` — it skips `@can_return_tuple`/`@auto_docstring` too) so it's fine, but worth a sentence: any path that depends on `capture_outputs` for `output_hidden_states=True` will silently break. None of the SFT path uses it; eval/debug paths might.

**Recommendation:** restructure §3–§6 to be explicit about the three-level patch, or attach `per_step_hidden_states` to a mutable buffer on the `Qwen3_5TextModel` instance (`self._last_per_step_hiddens = [...]`) and read it from `compute_loss` after the forward returns. The buffer approach is uglier but avoids re-implementing two forwards.

---

## 3. Critical — pre-norm + residual accumulation, no sandwich-norm

The plan correctly flags this in "Open questions" but underestimates how bad it can get. `Qwen3_5DecoderLayer` (modeling_qwen3_5.py:841) is **standard pre-norm with residuals outside the norms**:

```python
hidden_states = residual + sublayer(input_layernorm(hidden_states))
hidden_states = residual + mlp(post_attention_layernorm(hidden_states))
```

When you loop the stack T=4 times, the residual stream accumulates `T × num_layers` worth of sublayer outputs without any external normalization. For a 36-layer 4B model at T=4 that's ~288 stacked residual additions before the final `self.norm`. Activation magnitudes routinely blow up in this regime; the Ouro paper specifically calls out **sandwich-norm** as the stability fix that made T>2 work, and they still cap at T=4.

The plan's mitigation ("smaller `T_max` (≤4), conservative LR, watch for loss spikes — fall back to `T_max=2` if needed") is too optimistic. Practical mitigations to consider:

- Insert an extra `RMSNorm` between loop iterations (small, new param, can ship in `LoopGate`).
- Or, scale the residual contribution per loop by `1/sqrt(t)` on iterations t>1.
- Or, start at T=2 unconditionally for the smoke test in the plan's Step 1 instead of T=4.

Without one of these, expect the first run to NaN/Inf and waste a day.

---

## 4. Major — gate must be attached *before* `get_peft_model`, and excluded from LoRA targeting

`src/train/train_sft.py:23` defines `find_target_linear_names`, which **walks every `nn.Linear` in the model and adds it to LoRA targets** unless excluded by name match in `lora_namespan_exclude` (set on line 105 from a CLI arg). `LoopGate.gate` is `nn.Linear(d_model, 1)` — without an exclusion, PEFT will wrap it as a LoRA target, which is wrong (we want the full gate weight to train, not a low-rank update).

Also, the plan's §5 says "After `from_pretrained(...)`, attach `LoopGate`" — but `get_peft_model` is called later in `train_sft.py:182`. The order has to be:

1. `from_pretrained` → 2. patch forward → 3. attach `LoopGate` → 4. add `_loop_gate` to `lora_namespan_exclude` → 5. `get_peft_model` → 6. set `requires_grad=True` on gate params (PEFT will have frozen them by default, since they're not LoRA).

The plan's Step 7 sets `requires_grad=True` after the fact, which is right, but only works if step 4 is also done. Without step 4, the gate's Linear is replaced by `LoraLinear(gate, gate_lora_A, gate_lora_B)`; setting `requires_grad=True` on `_loop_gate.*` params won't bring back the original (now wrapped) weight matrix as a separately-trainable param.

Add to the plan:

```python
# in train_sft.py before get_peft_model
if training_args.loop_enable:
    training_args.lora_namespan_exclude += ["_loop_gate"]
```

---

## 5. Major — PEFT name-mangling will rename the gate path; verify the save filter

After `get_peft_model`, the gate's parameter name becomes something like `base_model.model.model.language_model._loop_gate.gate.weight`. `get_peft_state_non_lora_maybe_zero_3` (train_utils.py:50) keeps everything without `"lora_"` in the name and `requires_grad=True`. The gate clears that filter, so it will be saved into `non_lora_state_dict.bin`. Good.

But: this filter also picks up **every other `requires_grad=True` non-LoRA param** — e.g. if `freeze_vision_tower=False` someday, the entire vision tower lands in `non_lora_state_dict.bin`. With `loop_enable` we should be enforcing `freeze_vision_tower=True` and `freeze_merger=True` anyway (the plan implies SFT-first; make this explicit). Add a defensive check.

Also, `model.config.to_json_file` on `_save_checkpoint` (sft_trainer.py:179) only writes the merged config. The plan's §5 says to "stamp `loop_t_max` etc. onto `model.config`". Note that `Qwen3_5ForConditionalGeneration.config` is `Qwen3_5Config`, but the patched `Qwen3_5TextModel.forward` reads `self.config` which is the **inner `Qwen3_5TextConfig`** (modeling_qwen3_5.py:1303). So either:

- Stamp on both, or
- Stamp on the inner text config and read it from there everywhere (cleanest).

The plan reads `self.config.loop_t_max` inside the patched `Qwen3_5TextModel.forward` (§3 step 2). If you only stamp the outer config, that read returns `AttributeError`.

---

## 6. Major — per-step logit memory budget is worse than the plan suggests

The plan's open question lists this but the numbers deserve to be concrete. For Qwen3.5-4B (vocab ~152k, hidden 2560), at seq_len=4096, batch=1, bf16, `T_max=4`:

- Per-step logits tensor: `4096 × 152000 × 2 bytes = ~1.2 GB` per step.
- Four steps held simultaneously for the expected-CE sum: `~4.8 GB` per sample.
- Plus the per-step CE tensor of shape `[T, B, S]` and the `p_phi` tensor of the same shape.
- Plus `T_max=4` copies of the post-`self.norm` hidden state if you stash them: `4096 × 2560 × 2 × 4 = ~80 MB` per sample (negligible).

At batch=4 you're at ~20 GB just for logits, before activations and grads. The plan's mitigations (mask first, checkpoint head, readout only at `t∈{1,T_max}` early) are all correct, but at least one of them needs to be in the *Step 2 smoke test*, not deferred. Recommend: start with `--max_seq_length 1024` and label-mask the logits before computing CE (`logits = logits[label_mask]; targets = targets[label_mask]`) for the overfit run.

---

## 7. Major — gradient checkpointing claim is incorrect / double-counts

§3 says: *"Each loop iteration must be its own checkpoint segment so we don't keep T copies of activations. Wrap each pass through `self.layers` in `torch.utils.checkpoint.checkpoint`."*

`Qwen3_5DecoderLayer` is already a `GradientCheckpointingLayer` subclass (modeling_qwen3_5.py:828). With `--gradient_checkpointing True`, every layer is *already* checkpointed. Wrapping the outer loop in `torch.utils.checkpoint.checkpoint` on top of that gives you nested checkpointing: the outer `checkpoint` recomputes the inner forward (which itself does another `checkpoint` pass per layer). PyTorch's autograd allows it, but:

- Recompute cost roughly doubles vs. a single level of checkpointing.
- The plan's headline ("won't keep T copies of activations") is already true for the inner layers; the additional outer checkpoint only saves the boundary hidden states (`h^(t)` for t=1..T-1), which are small (`B*S*D` each). Not nothing, but not "T copies".

The genuinely useful trick is **not** stashing `per_step_hidden_states` as detached tensors *and* expecting gradients through the LM at intermediate depths — those are conflicting. To keep grads through `lm_head(h^(t))`, the activation graph for steps 1..T-1 has to live in memory. Recommend the plan picks one path:

- **Path A (memory):** detach `h^(t)` for `t<T`, project to logits with `lm_head`, compute CE, multiply by `p_phi` (which carries grad through the gate). Loss flows to gate only at intermediate steps; LM gets gradient only from step T's CE term. Cheap, but the LoRA never learns to be a good readout at intermediate steps.
- **Path B (signal):** keep all T graphs alive. Then nested checkpointing actually helps. Pay the memory.

Pick one in the plan and own the trade-off.

---

## 8. Moderate — `Qwen3_5TextModel.forward` already does per-loop work that you'd waste

Look at modeling_qwen3_5.py:1318–1388. Inside the existing forward, the following are computed once and reused across the layer loop:

- `causal_mask` and `linear_attn_mask` — both already cached, both per-layer-type.
- `position_embeddings = self.rotary_emb(...)` — already once.
- `cache_position`, `text_position_ids` slicing — already once.

The plan says "Runs the existing setup (embeddings, position ids, mask construction, rotary embeddings) once" — good, and the existing forward already does it once. So the looped variant is essentially: copy the body, wrap the `for layer in self.layers` block in an outer `for t in range(T)` loop. That's a smaller change than the plan implies. Worth being explicit about, because it reduces the surface area for bugs.

One subtlety: line 1370 selects mask per layer type (`linear_attn_mask if layer_type == "linear_attention" else causal_mask`). The looped variant has to keep this branch inside the loop body — easy to drop.

---

## 9. Moderate — Liger guard is already enforced upstream

§7 says: *"Add validation: refuse `--use_liger_kernel True` with `--loop_enable True`."* Good, but `train_sft.py:137-141` already auto-disables Liger for `qwen3_5` and `qwen3_5_moe`:

```python
if training_args.use_liger_kernel and model.config.model_type in {"qwen3_5", "qwen3_5_moe"}:
    rank0_print(f"Disabling Liger kernel for unsupported model_type: {model.config.model_type}")
    training_args.use_liger_kernel = False
```

So the validation is moot for the target model, but harmless. Drop or note it as defensive.

---

## 10. Moderate — Stage-1 loss masking is under-specified

§2 says: *"Mask out IGNORE_INDEX positions before summing"* for the per-step CE. Good. But the entropy regularizer term `H(p_phi)` is also computed per-position in the paper's formulation. If you compute `H` over all positions (including the `IGNORE_INDEX` prompt tokens), the gate is being pushed toward a uniform exit distribution on positions where there's no task signal — that's wasted gradient at best, harmful at worst (the gate has to learn to ignore prompt positions too, which is a different objective than learning when to halt computation for answer generation).

Tighten:

```python
mask = (labels != IGNORE_INDEX).float()           # [B, S]
expected = ((p * ce).sum(0) * mask).sum() / mask.sum()
H = (-(p * (p + 1e-9).log()).sum(0) * mask).sum() / mask.sum()
return expected - beta * H
```

Mention in the plan; it's not in §2's bulleted contract.

---

## 11. Minor — `LoopGate` initialization

§5 claims *"Linear with small init + zero bias → σ ≈ 0.5"*. Default `nn.Linear` init is Kaiming-uniform with bound `sqrt(1/fan_in)` → for `d_model=2560`, bound ≈ 0.0198. Pre-sigmoid logit on a normalized hidden state (post-RMSNorm, std~1) sums `d_model` independent uniform contributions: std ≈ `sqrt(d) × 0.0198 / sqrt(3)` ≈ 0.58. So `sigmoid(N(0, 0.58²))` is centered at 0.5 but with a wide spread (~ [0.30, 0.70]). Not quite "≈ 0.5".

For a tight prior at λ=0.5, **zero-init `gate.weight` and `gate.bias`** explicitly. Then the gate output is exactly 0.5 for every position at step 0, and the exit distribution starts uniform-with-geometric-tail rather than already biased.

```python
class LoopGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
```

---

## 12. Minor — DeepSpeed ZeRO + repeated `lm_head` reads

With ZeRO-3, `lm_head.weight` is sharded. Calling `model.lm_head(h^(t))` for `t=1..T_max` triggers T parameter gathers per training step. With `T_max=4` and a 4B model, that's 4× the all-gather traffic on `lm_head` per step. Not catastrophic (lm_head is one matmul among many), but worth measuring. If it shows up in profiles, gather once via `with deepspeed.zero.GatheredParameters(model.lm_head.weight): ...` around the per-step loop.

---

## 13. Minor — `_qwen3_5_mixed_modality_forward_impl` signature has no `**kwargs` for new args

The existing impl (monkey_patch_forward.py:86) takes named kwargs and `**kwargs: Unpack[TransformersKwargs]`. When you start propagating per-step state, prefer attaching it to a side-channel attribute on `self.language_model` (e.g. `self.language_model._last_loop_state = (per_step_hidden_states, per_step_lambdas)`) rather than threading new keys through `Unpack[TransformersKwargs]` — Unpack is a typed dict and adding new keys means widening the type. The trainer's `compute_loss` can then read `unwrap_model(model).model.language_model._last_loop_state`.

---

## 14. Minor — Resume from Stage 1 → Stage 2 is non-trivial

§9 says: *"Stage-2 follow-up script: same but `--loop_stage 2 --loop_gate_only True --learning_rate 1e-4`, resuming from Stage 1's output dir."* Stage 1 saves LoRA adapters + `non_lora_state_dict.bin` (containing the gate). Stage 2 needs both:

1. Load base model.
2. Re-apply patches.
3. Attach a fresh `LoopGate`.
4. Wrap with PEFT, load Stage-1 LoRA adapter (via `PeftModel.from_pretrained` or `set_peft_model_state_dict`).
5. Load `non_lora_state_dict.bin` into the model with `strict=False` so the gate weights land in `_loop_gate.gate.weight`.

`trainer.train(resume_from_checkpoint=True)` won't do any of this — it expects HF-style full-checkpoint resume. The plan should add a small "warm start" path (likely a CLI flag like `--load_from_loop_checkpoint <dir>`) and show it in the Stage-2 script.

---

## 15. Minor — Inference-time gate distribution shift

Even before the deferred KV-cache work, there's a basic train/inference gap: training uses `use_cache=False`, prefill uses `use_cache=True`. For full-attention layers this is irrelevant (same math). For the `Qwen3_5GatedDeltaNet` linear-attention layers, `chunk_gated_delta_rule` (training, no cache) and `recurrent_gated_delta_rule` (inference, decoding step) are different kernels with different numerical behavior. The gate is trained on the chunk-kernel hidden state distribution and used at inference on the recurrent-kernel distribution. Worth flagging as an open question alongside the KV cache discussion in §3, not as a separate problem to solve later but as a measurement to take during step-5 validation.

---

## 16. Nit — terminology

The plan says "Qwen3.5" throughout. The HF model_type is `qwen3_5` (and `qwen3_5_moe` for the MoE variant) and the `Qwen3.5-4B` checkpoint name is what the script uses. That's fine, but be clear in the README that "Qwen3.5" here means `qwen3_5` (the dense or MoE 'Next' family), not "Qwen 3 → Qwen 3.5" version bumps. Future readers will be confused.

---

## Suggested order-of-work edit

The plan's order is broadly right. One change: between current-Step-1 (skeleton smoke test) and current-Step-2 (Stage-1 loss path), insert a **norm-stability check**: run a single forward at T=4 on a real prompt and log `||hidden_states||` after each loop iteration. If you see >2× growth per loop, stop and fix activation drift before training anything. This costs 15 minutes and saves a day.

---

## Summary

The math, the loss formulations, the staging, and the high-level grafting strategy are all sound — this is recognizably the Ouro recipe correctly adapted for SFT. The implementation plan however assumes a simpler model architecture than Qwen3.5 actually has (hybrid linear/full attention; pre-norm rather than sandwich-norm; lm_head one wrapper above the looped stack; PEFT will eat the gate's Linear), so several "small wiring details" sections will turn into rewrites once you start. Issues 1, 2, 3, 4 each have nontrivial design implications and should be resolved in the plan before any code is written.
