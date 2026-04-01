import numpy as np
import torch
import torch.nn.functional as F

def choose_k_by_ratio(S: np.ndarray, c: float = 10.0) -> int:
    """
    S: 1D array of singular values sorted descending (as returned by np.linalg.svd).
    Returns smallest k >= 1 satisfying sum_{i<=k} S_i >= c * sum_{i>k} S_i.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.size == 0:
        return 0

    total = S.sum()
    if total <= 0:
        return 0

    target = (c / (c + 1.0)) * total  # same inequality rearranged
    prefix = np.cumsum(S)
    # first index where prefix >= target
    k0 = int(np.searchsorted(prefix, target, side="left"))  # 0-based
    k = max(1, min(S.size, k0 + 1))  # convert to 1-based k and clamp
    return k


def compute_feats_eigen(features, allowed_classes=None, min_class_count=0):
    eig_dict = {}
    keys = sorted(features.keys()) if allowed_classes is None else allowed_classes
    for c in keys:
        if c not in features:
            continue
        X = features[c]["feats"].astype(np.float32)
        n = int(X.shape[0]) if X.ndim == 2 else 0
        if n < min_class_count:
            continue  # <-- hard constraint
        _, S, v = np.linalg.svd(X, full_matrices=False)

        S = S.astype(np.float64)
        S1 = float(S[0]) if S.size > 0 else 0.0
        S2 = float(S[1]) if S.size > 1 else 0.0
        gap = float((S1 - S2) / (S1 + 1e-12)) if S1 > 0 else 0.0
        s1_ratio = float(S1 / (S.sum() + 1e-12)) if S.sum() > 0 else 0.0

        approx_k = choose_k_by_ratio(S, c=10)
        if approx_k == 0:
            continue
        v_r = v[0].astype(np.float32)
        v_n = v[approx_k:].astype(np.float32)
        v_r = v_r / (np.linalg.norm(v_r) + 1e-12)
        v_n = v_n / (np.linalg.norm(v_n, axis=1, keepdims=True) + 1e-12)
        eig_dict[int(c)] = {
            "n": n,
            "Vr": v_r,
            "Vn": v_n,
            "S1": S1,
            "S2": S2,
            "gap": gap,
            "s1_ratio": s1_ratio
            }
    return eig_dict


@torch.no_grad()
def extract_backbone_features(model, loader, device):
    model.eval()
    by_class = {}

    for x, y_noisy, y_clean, idx in loader:
        x = x.to(device, non_blocking=True)
        y_noisy = y_noisy.to(device, non_blocking=True).long()

        logits, z = model(x, return_feats=True)
        losses = F.cross_entropy(logits, y_noisy, reduction="none")
                                 
        y_noisy = y_noisy.detach().cpu().long()
        y_clean = y_clean.detach().cpu().long()
        idx = idx.detach().cpu().long()
        losses = losses.detach().cpu().float()
        z = z.detach().cpu().float()


        for i in range(z.size(0)):
            c = int(y_noisy[i].item())
            if c not in by_class:
                by_class[c] = {"feats": [], "idx": [], "y_noisy": [], "y_clean": [], "loss": []}
            by_class[c]["feats"].append(z[i:i+1])
            by_class[c]["idx"].append(idx[i:i+1])
            by_class[c]["y_noisy"].append(y_noisy[i:i+1])
            by_class[c]["y_clean"].append(y_clean[i:i+1])
            by_class[c]["loss"].append(losses[i:i+1])

    for c, rec in by_class.items():
        by_class[c] = {
            "feats":   torch.cat(rec["feats"], 0).numpy(),
            "idx":     torch.cat(rec["idx"], 0).numpy(),
            "y_noisy": torch.cat(rec["y_noisy"], 0).numpy(),
            "y_clean": torch.cat(rec["y_clean"], 0).numpy(),
            "loss":    torch.cat(rec["loss"], 0).numpy(),
        }
    return by_class