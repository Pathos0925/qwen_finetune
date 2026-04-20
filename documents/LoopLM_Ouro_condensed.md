# Scaling Latent Reasoning via Looped Language Models (Ouro)

Condensed, implementation-focused summary of arXiv:2510.25741 (ByteDance Seed et al., Nov 2025).

---

## 1. Core idea

A **Looped Language Model (LoopLM)** is a decoder-only Transformer whose block stack is applied **t times with shared weights** per forward pass. The number of recurrent steps `t ∈ {1, ..., T_max}` is chosen adaptively per input by a learned **exit gate**, letting easy inputs halt early and hard inputs compute deeper. Loop depth becomes a **third scaling axis**, alongside parameters and data.

Released models ("Ouro"): 1.4B and 2.6B params, pre-trained on 7.7T tokens, match 4B–8B standard transformers on most benchmarks (≈2–3× parameter efficiency).

---

## 2. Architecture

Standard decoder-only Transformer, plain and unmodified except for weight tying and an exit gate.

| Component | Choice |
|---|---|
| Attention | Multi-Head Attention + **RoPE** |
| FFN | **SwiGLU** |
| Norm | **RMSNorm, sandwich placement** (one before attention, one before FFN) |
| Tokenizer | SmolLM2, vocab = 49,152 |
| Positional | RoPE (base 10K → 40K → 1M across stages) |

| Model | Params | Layers L | d_model | T_max |
|---|---|---|---|---|
| Ouro 1.4B | 1.4B | 24 | 2048 | 4 |
| Ouro 2.6B | 2.6B | 48 | 2048 | 4 |

The 2.6B is "upcycled" from the 1.4B by duplicating its 24 layers to 48 — this is smooth for LoopLMs because weights are already shared across iterations.

### Forward pass

Let `M^L = T_θ_L ∘ ... ∘ T_θ_1` be one pass through the L-layer stack. A looped model with t recurrent steps is:

```
F^(t)(x) = lmhead( M^L ∘ M^L ∘ ... ∘ M^L (t times)  ∘  emb(x) )
```

`t = 1` recovers a vanilla transformer. At each step `t`, the hidden state `h^(t)` can be projected through `lmhead` to produce token logits, giving a per-step CE loss `L^(t)`.

### Exit gate (adaptive halting)

A small linear head on top of the final-layer hidden state produces a per-step **instantaneous exit probability**:

```
λ_t(x) = σ( Linear_φ( h^(t) ) )  ∈ (0, 1)
```

Define survival and the exit-step distribution:

```
S_t(x) = ∏_{j=1..t} (1 - λ_j(x)),    S_0 = 1

p_φ(t | x) =
  λ_t(x) · S_{t-1}(x)          for t = 1, ..., T_max − 1
  S_{T_max − 1}(x)              for t = T_max

CDF(n | x) = 1 − ∏_{j=1..n} (1 − λ_j(x))
```

**Q-exit inference:** pick threshold `q ∈ [0, 1]` and halt at the first `t` where `CDF(t) ≥ q`. Smaller `q` = earlier exits (less compute). `q` is a deployment-time knob.

---

## 3. Training objectives

Training has two stages for the gate.

### Stage I — joint LM + gate with entropy regularization

```
L = Σ_{t=1..T_max} p_φ(t | x) · L^(t)   −   β · H(p_φ(·|x))
```

The first term is expected task loss marginalized over exit steps; the second is an entropy bonus that prevents the gate from collapsing onto `t = T_max`.

Equivalently (ELBO view with **uniform prior** π_t = 1/T_max):

```
L_ELBO = Σ p_φ(t|x) L^(t)  +  β · KL( p_φ(·|x) ‖ uniform )
       = Σ p_φ(t|x) L^(t)  −  β · H(p_φ)  + const
```

The uniform prior is depth-unbiased — it decouples exit decisions from any compute preference. Geometric or Poisson priors bias toward early halting; the authors found uniform works better in this setting.

**β schedule:** 0.1 initially, reduced to 0.05 after Stage 1a to ease gradient conflict with task loss.

### Stage II — focused adaptive gate training

Freeze the LM, train only the gate `φ`. Build a supervised "ideal continuation" label from realized loss improvements:

```
I_i^(t) = max(0,  L_{i,stop}^{(t-1)}  −  L_{i,stop}^{(t)})       # detached
w_i^(t) = σ( k · (I_i^(t) − γ) ),   k = 50.0,   γ = 0.005
```

`w → 1` means "keep looping is still helping"; `w → 0` means "gains stalled, exit now". Gate loss is BCE between the predicted *continuation* probability `1 − λ` and label `w`:

```
L_adaptive^(t) = − (1/M) Σ_i [ w_i^(t) log(1 − λ_i^(t))  +  (1 − w_i^(t)) log(λ_i^(t)) ]
L_adaptive     = (1 / T_max) Σ_{t=2..T_max} L_adaptive^(t)
```

This simultaneously penalizes *underthinking* (exiting when improvement is still real) and *overthinking* (continuing when gains have stalled).

---

## 4. Pre-training recipe (7.7T tokens, 4 stages)

Shared optimizer: **AdamW** (β1=0.9, β2=0.95), weight decay 0.1, grad clip 1.0. WSD-style LR schedule.

| Stage | Name | Seq len | Batch | LR (final) | Schedule | Tokens | T (loops) | β (KL) | RoPE base |
|---|---|---|---|---|---|---|---|---|---|
| 1a | Pre-train I | 4K | 4M→8M | 3e-4 | Constant | 3T | **8** | 0.1 | 10K |
| 1b | Pre-train II | 4K | 8M | 3e-4 | Constant | 3T | **4** | 0.1 | 10K |
| 2 | CT Annealing | 16K | 8M | 3e-5 | Cosine decay | 1.4T | 4 | 0.05 | 40K |
| 3 | LongCT | 64K | 8M | 3e-5 | Constant | 20B | 4 | 0.05 | 1M |
| 4 | Mid-training | 32K | 8M | 1e-5 | Cosine decay | 300B | 4 | 0.05 | 1M |

**Key stability lessons:**
- 8 recurrent steps caused loss spikes / gradient oscillation — **reduce to 4**.
- Batch size scaling up to 8M tokens is needed; recurrence inflates gradient variance.
- Recurrent models need **smaller LR** than param-matched transformers.
- 2.6B was produced by **upcycling** (duplicating) the 24 layers of 1.4B to 48; shared recurrent weights make this painless.
- **Reducing β from 0.1 to 0.05** after early training lessens task/KL gradient conflict and lets the gate explore depths.

### Data composition

Stage 1 (6T tokens total): Nemotron-CC 73.4%, MAP-CC 13.0%, Ultra-FineWeb-zh 2.0%, OpenCoder-pretrain 7.5%, MegaMath-web 4.1%.

Stage 2 (1.4T): high-quality Nemotron-CC 66.5%, Nemotron-CC-Math-v1 15%, MegaMath-HQ 4.6%, code & SFT mixes ≈14%.

Stage 3: 20B from the ProLong 64K subset.

Stage 4: 90B tokens from an SFT-style mix (20+ datasets, decontaminated, ChatML-formatted) plus replay: 30B from Stage 1 + 180B from Stage 2 ≈ 300B effective.

---

## 5. Supervised fine-tuning (→ Ouro-Thinking)

8.3M examples, 2 epochs, max seq 32K. Optimizer Adam, lr = 2e-5 cosine, β = (0.9, 0.95). Uses LlamaFactory codebase.

| Domain | Sources | Size |
|---|---|---|
| Math | OpenThoughts3, AceReason-1.1-SFT | 3.5M |
| Code | AceReason-1.1-SFT, OpenCodeReasoning, Llama-Nemotron-PT, OpenThoughts3 | 3.2M |
| Science | OpenThoughts3, Llama-Nemotron-PT | 808K |
| Chat | OO1-Chat-747K, DeepWriting-20K | 767K |

**RL notes:** Exploratory RLVR (DAPO, GRPO) on DAPO-17K did not improve over SFT. Cause: vLLM/SGLang assume a fixed execution path, which breaks LoopLM's dynamic depth. Off-policy rollouts (rollout at full depth, compute loss at early depth) and fixed-4-round RL both failed to surpass SFT.

---

## 6. Inference

### Early-exit strategies (evaluated on MMLU)

1. **Static exit** — always exit at fixed step (1–4). Monotone improvement but wasteful.
2. **Hidden-state Δ threshold** — exit when `‖h_t − h_{t−1}‖_2 < ε`. Competitive, within 1–2% of the trained gate at matched budgets.
3. **Q-exit with entropy-trained gate** (Stage I only) — better than static.
4. **Q-exit with Stage-II-trained gate** — **best** at every compute budget; ≈2–3% better than (3) at matched average depth.

Typical operating point: at avg ~2.5 recurrent steps, Stage-II-trained gate reaches ~66% MMLU vs ~64% for Stage-I-only gate.

### KV cache sharing (4× memory reduction during decoding)

Per-step KV caches would cost 4× memory at T=4. Empirically:

| Strategy | GSM8K | MATH-500 | Memory |
|---|---|---|---|
| Full (4 caches) | 78.92 | 82.40 | 1× |
| First-step cache only | 18.73 | 8.43 | 0.25× (BROKEN) |
| **Last-step cache only** | **78.85** | **80.40** | **0.25×** |
| Averaged caches | 78.73 | 78.52 | 0.25× |

**Rules of thumb:**
- **Prefilling:** must keep all T caches; sharing costs >10 GSM8K points.
- **Decoding:** reuse the **last step's** KV cache only — negligible quality loss, 4× memory saved. The final recurrent step's representations are most informative for subsequent tokens.

### Step extrapolation

Even though trained at T = 4, Ouro models benefit from T = 5..8 at inference on some tasks (e.g., safety improves monotonically). Agreement saturates for T > 4 (the answer converges to a fixed point).

---

## 7. Why it works (empirical mechanisms)

- **Knowledge storage is unchanged.** Controlled bioS-style experiments show ≈2 bits/parameter for both looped and non-looped models. Looping does *not* increase raw capacity.
- **Knowledge manipulation improves.** On fact composition, multi-hop QA, and the Mano knowledge-manipulation benchmark, LoopLMs are substantially better at composing stored facts.
- **Theoretical intuition.** A looped transformer with O(log D) loops can solve reachability on a knowledge graph of diameter D, versus O(D) for continuous CoT and O(n²) for discrete CoT. Parameter sharing also appears to restrict the hypothesis class and improve sample complexity on tasks with recursive structure.
- **Safety improves with depth.** Harmfulness on HEx-PHI drops monotonically as T increases — including extrapolated T > 4. PCA on last-token states shows benign vs. harmful prompts become more separable at deeper steps.
- **Faithfulness.** Linear probes on intermediate states (Quora Question Pairs) show answers **change across loops** (step 2 ↔ step 4 agreement ≈ 36%), i.e. the latent trajectory really updates the decision, unlike CoT traces which often post-hoc rationalize.

---

## 8. Minimal implementation sketch (PyTorch-style pseudocode)

```python
class LoopLM(nn.Module):
    def __init__(self, L, d, n_heads, vocab, T_max):
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(L)])  # SwiGLU, RoPE, sandwich RMSNorm
        self.norm = RMSNorm(d)
        self.lmhead = nn.Linear(d, vocab, bias=False)
        self.gate = nn.Linear(d, 1)     # exit-gate scalar per position
        self.T_max = T_max

    def stack(self, h):
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(self, tokens, t_exec=None):
        h = self.emb(tokens)
        logits_per_step, lambdas = [], []
        T = t_exec or self.T_max
        for t in range(1, T + 1):
            h = self.stack(h)                        # shared weights reused
            logits_per_step.append(self.lmhead(self.norm(h)))
            lambdas.append(torch.sigmoid(self.gate(self.norm(h))).squeeze(-1))
        return logits_per_step, lambdas              # lists of length T

def exit_distribution(lambdas):                      # lambdas: list of [B, M]
    # returns p_phi[t] for t = 1..T_max
    surv = torch.ones_like(lambdas[0])
    probs = []
    for t, lam in enumerate(lambdas[:-1]):
        probs.append(lam * surv)
        surv = surv * (1 - lam)
    probs.append(surv)                               # mass for t = T_max
    return torch.stack(probs, dim=0)                 # [T_max, B, M]

def stage1_loss(logits_per_step, targets, lambdas, beta=0.1):
    ce = torch.stack([F.cross_entropy(lg.transpose(1,2), targets, reduction='none')
                      for lg in logits_per_step])    # [T, B, M]
    p  = exit_distribution(lambdas)                  # [T, B, M]
    expected = (p * ce).sum(0).mean()
    H = -(p * (p + 1e-9).log()).sum(0).mean()
    return expected - beta * H

def stage2_gate_loss(logits_per_step, targets, lambdas, k=50.0, gamma=0.005):
    # Freeze LM params outside this function.
    ce = [F.cross_entropy(lg.transpose(1,2), targets, reduction='none').detach()
          for lg in logits_per_step]                  # detached per-step loss
    loss = 0.0
    for t in range(1, len(ce)):
        I = torch.clamp(ce[t-1] - ce[t], min=0.0)
        w = torch.sigmoid(k * (I - gamma))
        lam = lambdas[t]                              # predicted exit prob at step t
        loss = loss + F.binary_cross_entropy(1 - lam, w)
    return loss / (len(ce) - 1)

@torch.no_grad()
def q_exit_generate(model, prefix, q=0.9):
    h = model.emb(prefix)
    surv = 1.0
    for t in range(1, model.T_max + 1):
        h = model.stack(h)
        lam = torch.sigmoid(model.gate(model.norm(h))).squeeze(-1)
        cdf = 1 - surv * (1 - lam)
        if cdf.min() >= q:                            # or position-wise exit policy
            break
        surv = surv * (1 - lam)
    return model.lmhead(model.norm(h))
```

**Notes for implementation.**
- Each recurrent step has its own KV cache during prefilling. During decoding, you can keep only the **last-step** cache; attention writes/reads against that single cache.
- Use bf16; conservative lr (≈half of what you'd use for a param-matched non-looped model). Start with T = 4, not 8.
- Use RMSNorm with sandwich placement (norm before *both* attention and FFN); this substantially helps stability in the deep unrolled graph.
- Entropy regularizer β in {0.1, 0.05} depending on stage; never 0 (otherwise gate collapses to T_max).
- Consider layer-duplication upcycling (duplicate each block) when scaling up — the shared-weight structure absorbs this cleanly.

---

## 9. Deployment bonuses (from Section 7.3)

- **Built-in speculative decoding.** Pair `Text(R_s)` (intermediate LM head readout) as draft with `Text(R_T)` as verifier. Both share parameters and KV prefix through step s.
- **Pre-emptive safety.** Screen draft distributions from `Text(R_s)` before streaming tokens, halt/reroute if a violation is detected.
- **Anytime generation.** Because `E[L^(t+1)] ≤ E[L^(t)]`, you can emit tokens from any step and keep refining — latency knob via `q`.

---

## 10. Summary of what you need to implement

1. Standard decoder Transformer with RoPE + SwiGLU + sandwich RMSNorm.
2. Share weights across `T_max` passes; keep per-step KV caches at train time.
3. Linear exit-gate head; per-step `lmhead` readouts.
4. **Stage I loss:** expected CE over exit distribution − β·H, β ≈ 0.05–0.1.
5. **Stage II loss:** freeze LM, BCE between `1−λ` and `σ(50·(ΔL − 0.005))`.
6. **Inference:** Q-exit on the CDF of the exit distribution; cache only last step during decode.
7. **Training stability:** T=4, AdamW (0.9, 0.95), wd 0.1, clip 1.0, WSD schedule, lower LR than baseline, batch ≥ 8M tokens, progressive seq length 4K → 16K → 64K → 32K.

That is enough to reproduce an Ouro-style LoopLM end-to-end.
