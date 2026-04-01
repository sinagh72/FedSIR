import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal, Optional

class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1.0, device="cuda"):
        super().__init__()
        cls_num = torch.tensor(cls_num_list, dtype=torch.float32, device=device)
        cls_p = cls_num / (cls_num.sum() + 1e-12)
        m_list = tau * torch.log(cls_p + 1e-12)
        self.m_list = m_list.view(1, -1)

    def forward(self, logits, target):
        logits_m = logits + self.m_list
        return F.cross_entropy(logits_m, target)


class LA_KD(nn.Module):
    def __init__(self, cls_num_list, tau=1.0, device="cuda"):
        super().__init__()
        cls_num = torch.tensor(cls_num_list, dtype=torch.float32, device=device)
        cls_p = cls_num / (cls_num.sum() + 1e-12)
        m_list = tau * torch.log(cls_p + 1e-12)
        self.m_list = m_list.view(1, -1)

    def forward(self, logits, target, soft_target, w_kd: float):
        logits_m = logits + self.m_list
        log_pred = torch.log_softmax(logits_m, dim=-1)
        log_pred = torch.where(torch.isinf(log_pred), torch.zeros_like(log_pred), log_pred)
        kl = F.kl_div(log_pred, soft_target, reduction="batchmean")
        nll = F.nll_loss(log_pred, target)
        return w_kd * kl + (1.0 - w_kd) * nll


class CE_KD(nn.Module):
    def forward(self, logits, target, soft_target, w_kd: float):
        log_pred = torch.log_softmax(logits, dim=-1)
        log_pred = torch.where(torch.isinf(log_pred), torch.zeros_like(log_pred), log_pred)
        kl = F.kl_div(log_pred, soft_target, reduction="batchmean")
        ce = F.cross_entropy(logits, target)
        return w_kd * kl + (1.0 - w_kd) * ce


LossMode = Literal["ce", "la"]
@dataclass
class LossConfig:
    mode: LossMode = "ce"
    cls_num_list: Optional[list] = None  # for LogitAdjust loss

def build_loss_fn(loss_cfg: LossConfig, device=None) -> nn.Module:
    if loss_cfg.mode == "ce":
        return nn.CrossEntropyLoss()

    if loss_cfg.mode == "la":
            return LogitAdjust(cls_num_list=loss_cfg.cls_num_list, device=device)

    raise ValueError(f"Unknown loss mode: {loss_cfg.mode}")

