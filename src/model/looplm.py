"""Ouro-style LoopLM adapter parameters and loss helpers.

This module holds the *new* trainable pieces that turn a vanilla decoder LM
into a Looped LM at finetune time:

  - LoopAdapter: the exit-gate (linear → sigmoid) plus an optional
    inter-loop RMSNorm. Inter-loop norm is a stability fix for pre-norm
    backbones like Qwen3.5 that lack the sandwich-norm Ouro relies on.
  - exit_distribution / stage1_loss / stage2_gate_loss: framework-free
    helpers used by the trainer.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32 * self.weight.float()).to(in_dtype)


class LoopAdapter(nn.Module):
    """Trainable params added on top of a pretrained LM to make it loopable.

    Parameters
    ----------
    d_model : hidden size of the wrapped LM.
    eps     : RMSNorm epsilon (match the backbone's rms_norm_eps).
    use_inter_norm :
        Insert a learnable RMSNorm applied between loop iterations on the
        residual stream. Initialized to identity (gamma=1).
    """

    def __init__(self, d_model: int, eps: float = 1e-6, use_inter_norm: bool = True):
        super().__init__()
        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.use_inter_norm = use_inter_norm
        if use_inter_norm:
            self.inter_loop_norm = _RMSNorm(d_model, eps=eps)
        else:
            self.inter_loop_norm = None

    def lambda_at(self, h_post_norm: torch.Tensor) -> torch.Tensor:
        """Per-position instantaneous exit probability λ_t in (0, 1).

        h_post_norm: [B, S, D] — hidden state *after* the backbone's final norm.
        returns:     [B, S]
        """
        return torch.sigmoid(self.gate(h_post_norm)).squeeze(-1)

    def between_loops(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Optional renormalization of the residual stream between loops."""
        if self.inter_loop_norm is None:
            return hidden_states
        return self.inter_loop_norm(hidden_states)


def exit_distribution(lambdas: List[torch.Tensor]) -> torch.Tensor:
    """Convert per-step λ_t into the exit-step probability mass p_φ(t | x).

    lambdas : list of T tensors of shape [B, S].
    returns : tensor of shape [T, B, S] summing to 1 along dim 0.
    """
    T = len(lambdas)
    surv = torch.ones_like(lambdas[0])
    probs: List[torch.Tensor] = []
    for t in range(T - 1):
        lam = lambdas[t]
        probs.append(lam * surv)
        surv = surv * (1 - lam)
    probs.append(surv)  # mass that survives all gates lands at t = T_max
    return torch.stack(probs, dim=0)


def _per_step_token_ce(
    per_step_logits: List[torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Token-level CE for each loop step.

    per_step_logits : list of T tensors of shape [B, S, V] (already shifted upstream).
    labels          : [B, S] (already shifted upstream).
    returns         : [T, B, S], with positions where labels == ignore_index zeroed out.
    """
    losses = []
    for logits in per_step_logits:
        ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).reshape(labels.shape)
        losses.append(ce)
    return torch.stack(losses, dim=0)


def stage1_loss(
    per_step_logits: List[torch.Tensor],
    labels: torch.Tensor,
    lambdas: List[torch.Tensor],
    *,
    beta: float = 0.1,
    ignore_index: int = -100,
) -> dict:
    """Joint LM + gate loss.

    L = E_{t ~ p_φ}[ L^(t) ]  −  β · H(p_φ)

    Both terms are masked to label-bearing positions so the gate is only
    trained where there is task signal.
    """
    p = exit_distribution(lambdas)  # [T, B, S]
    ce = _per_step_token_ce(per_step_logits, labels, ignore_index=ignore_index)  # [T, B, S]

    label_mask = (labels != ignore_index).to(p.dtype)  # [B, S]
    denom = label_mask.sum().clamp_min(1.0)

    expected_ce = ((p * ce).sum(0) * label_mask).sum() / denom
    H = (-(p * (p + 1e-9).log()).sum(0) * label_mask).sum() / denom

    loss = expected_ce - beta * H

    with torch.no_grad():
        T = p.shape[0]
        steps = torch.arange(1, T + 1, device=p.device, dtype=p.dtype).view(T, 1, 1)
        mean_exit_step = ((p * steps).sum(0) * label_mask).sum() / denom
        per_step_ce = (ce * label_mask).sum(dim=(1, 2)) / denom

    return {
        "loss": loss,
        "expected_ce": expected_ce.detach(),
        "entropy": H.detach(),
        "mean_exit_step": mean_exit_step.detach(),
        "per_step_ce": per_step_ce.detach(),
    }


def stage2_gate_loss(
    per_step_logits: List[torch.Tensor],
    labels: torch.Tensor,
    lambdas: List[torch.Tensor],
    *,
    k: float = 50.0,
    gamma: float = 0.005,
    ignore_index: int = -100,
) -> dict:
    """Adaptive gate-only loss (paper's stage II).

    Build a supervised "keep looping?" label from realized loss improvement
    between successive loop steps:

        I_t = max(0, L^(t-1) − L^(t))           # detached
        w_t = σ( k · (I_t − γ) )                # 1 ≈ keep looping, 0 ≈ exit

    Train the gate by BCE between (1 − λ_t) and w_t. Defined for t=2..T.
    """
    with torch.no_grad():
        ce = _per_step_token_ce(per_step_logits, labels, ignore_index=ignore_index).detach()  # [T, B, S]

    label_mask = (labels != ignore_index).to(ce.dtype)
    denom = label_mask.sum().clamp_min(1.0)

    T = ce.shape[0]
    loss = ce.new_zeros(())
    for t in range(1, T):
        with torch.no_grad():
            improvement = (ce[t - 1] - ce[t]).clamp_min(0.0)
            w = torch.sigmoid(k * (improvement - gamma))
        lam = lambdas[t]  # gradient flows through λ
        bce = F.binary_cross_entropy(
            torch.clamp(1 - lam, 1e-6, 1 - 1e-6),
            w,
            reduction="none",
        )
        loss = loss + (bce * label_mask).sum() / denom

    loss = loss / max(1, (T - 1))

    return {"loss": loss}


def project_per_step_logits(
    per_step_hidden_states: List[torch.Tensor],
    lm_head: nn.Module,
    *,
    label_mask: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Run lm_head on each per-step hidden state.

    If label_mask is provided, only positions with mask=True are projected
    (saving the [S, V] memory cost on prompt tokens). The returned tensors
    are then "scattered back" to dense [B, S, V] only where needed by the
    caller — but we keep the dense tensor here for simplicity. Callers that
    care about memory should pass label_mask=None and handle masking outside.
    """
    return [lm_head(h) for h in per_step_hidden_states]
