from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from model.loss import LossConfig, build_loss_fn
from configs import ExpConfig


@dataclass
class LogContext:
    round: int | None = None
    client: int | None = None
    stage: str | None = None 


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "sgd"   # "sgd" or "adamw"
    grad_clip: float = 0.0
    log_every: int = 50
    num_classes: Optional[int] = None


def _prox_term(model: nn.Module, global_params: dict, device: torch.device) -> torch.Tensor:
    """
    Computes sum ||w - w_global||^2 over trainable floating params.
    global_params: dict[name -> Tensor(cpu or gpu)]
    """
    prox = torch.zeros((), device=device)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not torch.is_floating_point(p):
            continue
        g = global_params[name].to(device)
        prox = prox + torch.sum((p - g) ** 2)
    return prox


def _select_y(batch, label_mode: str, allowed_idx_set=None, y_override_map=None):
    """
    Returns:
      x, y, idx  (tensors)
    If filtering removes the entire batch, returns (None, None, None).
    """
    if len(batch) == 2:
        x, y = batch
        # no idx available in this format
        return x, y, None

    x, y_noisy, y_clean, idx = batch[:4]
    y = y_noisy if label_mode == "noisy" else y_clean

    # no filtering / overriding
    if allowed_idx_set is None and y_override_map is None:
        return x, y, idx

    idx_np = idx.detach().cpu().numpy().astype(int)

    # filter by allowed indices
    if allowed_idx_set is not None:
        keep = np.array([i in allowed_idx_set for i in idx_np], dtype=bool)
    else:
        keep = np.ones_like(idx_np, dtype=bool)

    if not np.any(keep):
        return None, None, None

    x = x[keep]
    y = y[keep]
    idx_keep = idx[keep]
    idx_keep_np = idx_np[keep]

    # override labels (only for kept indices)
    if y_override_map is not None:
        y_new = np.array([y_override_map[int(i)] for i in idx_keep_np], dtype=np.int64)
        y = torch.from_numpy(y_new).to(y.device)

    return x, y, idx_keep


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
    label_mode: str = "noisy",
    ctx: Optional[LogContext] = None,
    loss_cfg: Optional[LossConfig] = None,
    seed: int = 42,
    epoch: int = 0,
    mu: float = 0.01,
    global_params: dict = None,
) -> Dict[str, float]:
    assert label_mode in ["noisy", "clean"]
    model.train()

    if loss_cfg is None:
        loss_cfg = LossConfig(mode="ce")

    loss_fn = build_loss_fn(loss_cfg, device=device)
    loss_meter, acc_meter, n_samples = 0.0, 0.0, 0

    prefix = []
    if ctx is not None:
        if ctx.round is not None:  prefix.append(f"R{ctx.round:02d}")
        if ctx.client is not None: prefix.append(f"C{ctx.client:03d}")
        if ctx.stage:              prefix.append(str(ctx.stage))

    pdesc = " ".join(prefix) + (": " if prefix else "")
    pbar = tqdm(loader, desc=f"{pdesc}Train({label_mode})", leave=False)

    rng = np.random.RandomState(seed + epoch)

    for batch in pbar:
        x, y, idx = _select_y(batch, label_mode)
        if x is None:
            continue

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()
        idx = idx.to(device, non_blocking=True) if idx is not None else None

        optimizer.zero_grad(set_to_none=True)

        logits = model(x, return_feats=False)

        loss = loss_fn(logits, y)
        # ----- Prox term -----
        if global_params and mu > 0:
            prox = _prox_term(model, global_params, device)
            loss = loss + 0.5 * mu * prox

        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bsz = int(y.size(0))
        n_samples += bsz
        loss_meter += float(loss.item()) * bsz
        acc_meter += (logits.argmax(dim=1) == y).float().mean().item() * bsz

        pbar.set_postfix(loss=f"{loss_meter/max(n_samples,1):.4f}",
                         acc=f"{acc_meter/max(n_samples,1):.4f}",
                         )

    return {
        "loss": loss_meter / max(n_samples, 1),
        "acc":  acc_meter / max(n_samples, 1),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    train_cfg: TrainConfig,
    cfg: ExpConfig,
    label_mode: str = "noisy",  
    ctx: Optional[LogContext] = None,
    loss_cfg: Optional[LossConfig] = None,
    mu: float = 0,
    global_params: dict = None,
) -> Tuple[nn.Module, Dict[str, float]]:
    assert label_mode in ["noisy", "clean"]

    if train_cfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=train_cfg.lr, momentum=train_cfg.momentum, weight_decay=train_cfg.weight_decay)
    elif train_cfg.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    elif train_cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {train_cfg.optimizer}")

    model.train()

    best_state = None
    best_metrics = {}

    for epoch in range(train_cfg.epochs):
        t0 = time.time()

        train_stats = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device,
            grad_clip=train_cfg.grad_clip, label_mode=label_mode, ctx=ctx, loss_cfg=loss_cfg, seed=cfg.seed, epoch=epoch,
            mu=mu, global_params=global_params, 
        )

        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_metrics = {"epoch": epoch, **train_stats}

        dt = time.time() - t0
        prefix = ""
        if ctx is not None:
            parts = []
            if ctx.round is not None:  parts.append(f"R{ctx.round:02d}")
            if ctx.client is not None: parts.append(f"C{ctx.client:03d}")
            if ctx.stage:              parts.append(str(ctx.stage))
            if parts:
                prefix = "[" + " ".join(parts) + "] "

        print(
            f"{prefix}"
            f"[E{epoch+1:03d}/{train_cfg.epochs:03d}] "
            f"train({label_mode}) loss={train_stats['loss']:.4f} acc={train_stats['acc']:.4f} | "
            f"{dt:.1f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    losses = []  # per-sample CE

    for batch in loader:
        if len(batch) == 2:
            x, y = batch
        else:
            x, y = batch[0], batch[2]

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long()

        logits = model(x)
        per_sample = F.cross_entropy(logits, y, reduction="none")  # [bs]
        pred = logits.argmax(dim=1)

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        losses.append(per_sample.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    losses = np.concatenate(losses)  # [N]
    return y_true, y_pred, losses
