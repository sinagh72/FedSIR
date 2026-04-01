import numpy as np

def _flatten_feats_dict(feats_by_class: dict):
    """Flatten your extract_backbone_features output into arrays."""
    idx_list, z_list, yno_list, ycl_list = [], [], [], []
    for c in sorted(feats_by_class.keys()):
        rec = feats_by_class[c]
        idx_list.append(rec["idx"].astype(np.int64).reshape(-1))
        z_list.append(rec["feats"].astype(np.float32))
        yno_list.append(rec["y_noisy"].astype(np.int64).reshape(-1))
        ycl_list.append(rec["y_clean"].astype(np.int64).reshape(-1))

    idx_all = np.concatenate(idx_list, 0)
    Z_all = np.concatenate(z_list, 0)
    y_noisy_all = np.concatenate(yno_list, 0)
    y_clean_all = np.concatenate(ycl_list, 0)
    return idx_all, Z_all, y_noisy_all, y_clean_all


def _mode_decision_from_scores(
    SR: np.ndarray,  # [N,C], larger is better
    SN: np.ndarray,  # [N,C], smaller is better
    *,
    mode: str,
    margin_thr=0.2,
    fill_with_original: bool,
    y_noisy_all: np.ndarray,
    ratio_eps: float = 1e-6,
    ratio_margin_thr: float = 0.0,
):
    N, C = SR.shape
    arg_sr = SR.argmax(axis=1)
    sr_max = SR[np.arange(N), arg_sr]

    SR_sorted = np.sort(SR, axis=1)
    sr_second = SR_sorted[:, -2] if C >= 2 else np.zeros_like(sr_max)
    sr_margin = sr_max - sr_second

    arg_sn = SN.argmin(axis=1)
    sn_min = SN[np.arange(N), arg_sn]

    agree = (arg_sr == arg_sn)
    disagree = ~agree

    # ratio: lower is better
    ratio = SN / (SR + ratio_eps)
    arg_ratio = ratio.argmin(axis=1)
    ratio_best = ratio[np.arange(N), arg_ratio]

    ratio_sorted = np.sort(ratio, axis=1)  # ascending
    ratio_second = ratio_sorted[:, 1] if C >= 2 else np.full_like(ratio_best, np.inf)
    ratio_margin = ratio_second - ratio_best

    y_pred = np.full((N,), -1, dtype=np.int64)
    method = np.full((N,), "", dtype=object)

    mode = mode.lower()

    if mode == "sr":
        y_pred[:] = arg_sr
        method[:] = "sr"

    elif mode == "sn":
        y_pred[:] = arg_sn
        method[:] = "sn"

    elif mode == "ratio":
        y_pred[:] = arg_ratio
        method[:] = "ratio"

    elif mode == "agree":
        y_pred[agree] = arg_sr[agree]
        method[agree] = "agree(sr=sn)"

    elif mode == "agree_then_sr":
        y_pred[agree] = arg_sr[agree]
        method[agree] = "agree(sr=sn)"
        y_pred[disagree] = arg_sr[disagree]
        method[disagree] = "sr_disagree"

    elif mode == "agree_then_sn":
        y_pred[agree] = arg_sr[agree]
        method[agree] = "agree(sr=sn)"
        y_pred[disagree] = arg_sn[disagree]
        method[disagree] = "sn_disagree"

    elif mode == "agree_then_ratio":
        y_pred[agree] = arg_sr[agree]
        method[agree] = "agree(sr=sn)"
        y_pred[disagree] = arg_ratio[disagree]
        method[disagree] = "ratio_disagree"

    elif mode == "agree_then_ratio_margin":
        y_pred[agree] = arg_sr[agree]
        method[agree] = "agree(sr=sn)"

        d = np.where(disagree)[0]
        if d.size > 0:
            conf_ratio = ratio_margin[d] > float(ratio_margin_thr)
            y_pred[d[conf_ratio]] = arg_ratio[d[conf_ratio]]
            method[d[conf_ratio]] = "ratio_conf"
            # leave low-confidence ones unlabeled for orig fill

    elif mode == "agree_then_margin":
        y_pred[agree] = arg_sr[agree]
        method[agree] = "agree(sr=sn)"

        d = np.where(disagree)[0]
        if d.size > 0:
            if np.isscalar(margin_thr):
                thr = float(margin_thr)
                conf_sr = sr_margin[d] > thr
            else:
                thr = np.asarray(margin_thr, dtype=np.float32)
                assert thr.shape[0] == C
                conf_sr = sr_margin[d] > thr[arg_sr[d]]

            y_pred[d[conf_sr]] = arg_sr[d[conf_sr]]
            method[d[conf_sr]] = "sr_conf"

            y_pred[d[~conf_sr]] = arg_sn[d[~conf_sr]]
            method[d[~conf_sr]] = "sn_fallback"

    else:
        raise ValueError(f"Unknown mode='{mode}'")

    if fill_with_original:
        unlabeled = (y_pred < 0)
        y_pred[unlabeled] = y_noisy_all[unlabeled]
        ul = np.where(unlabeled)[0]
        for i in ul:
            method[i] = (method[i] + "|orig") if method[i] else "orig"

    return {
        "y_pred": y_pred,
        "method": method,
        "sr_max": sr_max.astype(np.float32),
        "sr_margin": sr_margin.astype(np.float32),
        "sn_min": sn_min.astype(np.float32),
        "arg_sr": arg_sr,
        "arg_sn": arg_sn,
        "agree": agree,
        "ratio_best": ratio_best.astype(np.float32),
        "ratio_margin": ratio_margin.astype(np.float32),
        "arg_ratio": arg_ratio.astype(np.int64),
    }



def relabel_noisy_samples_from_clean_dict(
    *,
    feats_noisy: dict,
    eigs_clean_by_client: dict,     # cid -> eig_dict from compute_feats_eigen (per class)
    clean_cids: list,
    num_classes: int,
    weight_mode: str = "n",         # "n" or "n_gap" (only used for weighted_scores)
    mode: str = "agree",
    margin_thr=0.2,
    fill_with_original: bool = True,
    eps: float = 1e-12,
):
    """

    Returns dict similar to your relabel_noisy_samples plus diagnostics.
    """
    idx_all, Z_all, y_noisy_all, y_clean_all = _flatten_feats_dict(feats_noisy)
    N, D = Z_all.shape
    Z_hat = Z_all / (np.linalg.norm(Z_all, axis=1, keepdims=True) + eps)

    # ---- sanity: require each class appears in at least one clean client ----
    present_counts = np.zeros((num_classes,), dtype=np.int32)
    for cid in clean_cids:
        ed = eigs_clean_by_client[cid]
        for c in ed.keys():
            if 0 <= int(c) < num_classes:
                present_counts[int(c)] += 1
    missing = np.where(present_counts == 0)[0].tolist()
    if missing:
        raise ValueError(f"No clean client provides eig for classes: {missing}")

    
    agg = aggregate_eigs_projector_mean(
        eigs_clean_by_client=eigs_clean_by_client,
        clean_cids=clean_cids,
        num_classes=num_classes,
        K_noise=12,
        weight_mode=weight_mode,     # "n", "n_gap", "n_s1", "n_gap_s1"
        min_class_n=100,              # ignore tiny-n per class
        n_transform="linear",          # "sqrt" or "log1p" or "linear"
        strength_fallback=True,      # protect against bad aggregation
        strength_thr=0.03,           # tune/percentile; smaller => more trusting
        eps=eps,
    )

    # build vR_hat and compute SR
    vR = np.stack([agg[c]["Vr"] for c in range(num_classes)], axis=0).astype(np.float32)  # [C,D]
    vR_norm = np.linalg.norm(vR, axis=1) + eps
    vR_hat = vR / vR_norm[:, None]
    SR = np.abs(Z_hat @ vR_hat.T).astype(np.float32)  # [N,C]

    # compute SN
    SN = np.zeros((N, num_classes), dtype=np.float32)
    for c in range(num_classes):
        Vn = agg[c]["Vn"].astype(np.float32)  # [K,D]
        proj = Z_all @ Vn.T                   # [N,K]
        # proj = Z_hat @ Vn.T                   # [N,K]
        SN[:, c] = np.linalg.norm(proj, axis=1) / np.sqrt(max(1, Vn.shape[0]))

    dec = _mode_decision_from_scores(
        SR, SN,
        mode=mode,
        margin_thr=margin_thr,
        fill_with_original=fill_with_original,
        y_noisy_all=y_noisy_all,
    )

    if "agree" in dec:
        agree = dec["agree"]
        print(f"agree rate: {agree.mean():.4f}")

    if "arg_ratio" in dec and "ratio_margin" in dec:
        arg_ratio = dec["arg_ratio"]
        ratio_margin = dec["ratio_margin"]

        dis = ~dec["agree"]
        if dis.any():
            ratio_dis_acc = float((arg_ratio[dis] == y_clean_all[dis]).mean())
            print(f"ratio accuracy on disagreements: {ratio_dis_acc:.4f}")
            print(f"ratio margin mean on disagreements: {ratio_margin[dis].mean():.4f}")

    return {
        "idx": idx_all,
        "y_pred": dec["y_pred"],
        "method": dec["method"],
        "sr_max": dec["sr_max"],
        "sr_margin": dec["sr_margin"],
        "sn_min": dec["sn_min"],
        "y_noisy": y_noisy_all,
        "y_clean": y_clean_all,
        "arg_sr": dec["arg_sr"],
        "arg_sn": dec["arg_sn"],
        "agree": dec["agree"],
        "weight_mode": weight_mode,
        "agg_w_sum_per_class": np.array([agg[c].get("w_sum", 0.0) for c in range(num_classes)], dtype=np.float32),
        "agg_strength_per_class": np.array([agg[c].get("strength", 0.0) for c in range(num_classes)], dtype=np.float32),
        "agg_fallback_used": np.array([1 if agg[c].get("fallback_used", False) else 0 for c in range(num_classes)], dtype=np.int32),
        "agg_best_cid_per_class": np.array([agg[c].get("best_cid", -1) for c in range(num_classes)], dtype=np.int64),
    }
    
    
def aggregate_eigs_projector_mean(
    *,
    eigs_clean_by_client: dict,   # cid -> {c -> {"Vr","Vn","n","gap","s1_ratio"...}}
    clean_cids: list,
    num_classes: int,
    K_noise: int = 8,
    weight_mode: str = "n_gap_s1",     # "n", "n_gap", "n_s1", "n_gap_s1"
    min_class_n: int = 100,             # ignore tiny per-class sample counts
    n_transform: str = "linear",         # "sqrt" | "log1p" | "linear"
    strength_fallback: bool = True,    # if agg consensus weak, fallback to max_n per class
    strength_thr: float = 0.03,        # tune (e.g., 0.01~0.1)
    eps: float = 1e-12,
):
    """
    Projector-mean aggregation:
      M_r = sum_i w_i * (vr_i vr_i^T)
      M_n = sum_i w_i * (Vn_i^T Vn_i)

    vr_star = top eigenvector of M_r
    Vn_star = top-K eigenvectors of M_n

    Returns:
      agg[c] -> {
        "Vr": [D], "Vn": [K,D], "w_sum": float, "strength": float,
        "fallback_used": bool, "best_cid": int
      }

    Notes:
      - uses per-client per-class weights w_i[c] that can include n, gap, s1_ratio
      - optionally falls back to the max_n client if aggregated "strength" is too low
    """
    weight_mode = str(weight_mode).lower()
    n_transform = str(n_transform).lower()

    # -------------------------
    # infer D and validate keys
    # -------------------------
    D = None
    for cid in clean_cids:
        ed = eigs_clean_by_client.get(cid, {})
        if len(ed) == 0:
            continue
        any_c = next(iter(ed.keys()))
        rec = ed[any_c]
        if "Vr" in rec:
            D = int(np.asarray(rec["Vr"]).reshape(-1).shape[0])
            break
    if D is None:
        raise ValueError("No eigs provided by clean clients (missing 'Vr').")

    def _n_weight(n: float) -> float:
        n = max(float(n), 0.0)
        if n_transform == "log1p":
            return float(np.log1p(n))
        if n_transform == "linear":
            return float(n)
        # default sqrt
        return float(np.sqrt(n))

    # -------------------------
    # precompute per-class max_n fallback source
    # -------------------------
    best_cid = [-1] * num_classes
    best_n = [-1.0] * num_classes
    for cid in clean_cids:
        ed = eigs_clean_by_client.get(cid, {})
        for c in range(num_classes):
            if c not in ed:
                continue
            n_c = float(ed[c].get("n", 0.0))
            if n_c > best_n[c]:
                best_n[c] = n_c
                best_cid[c] = int(cid)

    agg = {}

    for c in range(num_classes):
        M_r = np.zeros((D, D), dtype=np.float64)
        M_n = np.zeros((D, D), dtype=np.float64)
        w_sum = 0.0

        # accumulate
        for cid in clean_cids:
            ed = eigs_clean_by_client.get(cid, {})
            if c not in ed:
                continue
            rec = ed[c]

            n = float(rec.get("n", 0.0))
            if n < float(min_class_n):
                continue

            vr = np.asarray(rec["Vr"], dtype=np.float64).reshape(-1)
            vr = vr / (np.linalg.norm(vr) + eps)

            Vn = np.asarray(rec["Vn"], dtype=np.float64)
            if Vn.ndim == 1:
                Vn = Vn[None, :]
            Vn = Vn[:min(int(K_noise), int(Vn.shape[0])), :]
            Vn = Vn / (np.linalg.norm(Vn, axis=1, keepdims=True) + eps)

            gap = float(rec.get("gap", 0.0))
            s1r = float(rec.get("s1_ratio", 1.0))  # if missing, don’t kill it

            n_w = _n_weight(n)

            if weight_mode == "n_gap_s1":
                w = n_w * max(0.0, gap) * max(0.0, s1r)
            elif weight_mode == "n_s1":
                w = n_w * max(0.0, s1r)
            elif weight_mode == "n_gap":
                w = n_w * max(0.0, gap)
            elif weight_mode == "n":
                w = n_w
            else:
                raise ValueError("weight_mode must be one of: 'n', 'n_gap', 'n_s1', 'n_gap_s1'")

            if w <= 0:
                continue

            M_r += w * np.outer(vr, vr)     # rank-1 projector
            M_n += w * (Vn.T @ Vn)          # subspace projector
            w_sum += w

        if w_sum <= 0:
            # no contributors for this class -> must fallback to best available
            cid_fb = best_cid[c]
            if cid_fb < 0:
                raise ValueError(f"class {c} has no clean eig provider at all.")
            rec = eigs_clean_by_client[cid_fb][c]
            vr_fb = np.asarray(rec["Vr"], dtype=np.float32).reshape(-1)
            vr_fb = vr_fb / (np.linalg.norm(vr_fb) + eps)
            Vn_fb = np.asarray(rec["Vn"], dtype=np.float32)
            if Vn_fb.ndim == 1:
                Vn_fb = Vn_fb[None, :]
            Vn_fb = Vn_fb[:min(K_noise, Vn_fb.shape[0]), :]
            Vn_fb = Vn_fb / (np.linalg.norm(Vn_fb, axis=1, keepdims=True) + eps)

            agg[c] = {
                "Vr": vr_fb.astype(np.float32),
                "Vn": Vn_fb.astype(np.float32),
                "w_sum": 0.0,
                "strength": 0.0,
                "fallback_used": True,
                "best_cid": int(cid_fb),
            }
            continue

        # normalize (scale only; eigenvectors unaffected, but keeps numbers stable)
        M_r /= (w_sum + eps)
        M_n /= (w_sum + eps)

        # consensus vr = principal eigenvector of M_r
        evals_r, evecs_r = np.linalg.eigh(M_r)
        order_r = np.argsort(evals_r)  # ascending
        vr_star = evecs_r[:, order_r[-1]]
        vr_star = vr_star / (np.linalg.norm(vr_star) + eps)

        # strength = top1 - top2 eigen gap (agreement indicator)
        if evals_r.size >= 2:
            strength = float(evals_r[order_r[-1]] - evals_r[order_r[-2]])
        else:
            strength = float(evals_r[order_r[-1]])

        # consensus Vn = top-K eigenvectors of M_n
        evals_n, evecs_n = np.linalg.eigh(M_n)
        order_n = np.argsort(evals_n)[::-1]  # descending
        K = min(int(K_noise), D)
        Vn_star = evecs_n[:, order_n[:K]].T  # [K,D]
        Vn_star = Vn_star / (np.linalg.norm(Vn_star, axis=1, keepdims=True) + eps)

        agg[c] = {
            "Vr": vr_star.astype(np.float32),
            "Vn": Vn_star.astype(np.float32),
            "w_sum": float(w_sum),
            "strength": float(strength),
            "fallback_used": False,
            "best_cid": int(best_cid[c]),
        }

    # -------------------------
    # optional: fallback if consensus is weak
    # -------------------------
    if strength_fallback:
        for c in range(num_classes):
            if c not in agg:
                continue
            if agg[c]["fallback_used"]:
                continue
            if float(agg[c]["strength"]) < float(strength_thr):
                cid_fb = best_cid[c]
                if cid_fb < 0:
                    continue
                rec = eigs_clean_by_client[cid_fb][c]
                vr_fb = np.asarray(rec["Vr"], dtype=np.float32).reshape(-1)
                vr_fb = vr_fb / (np.linalg.norm(vr_fb) + eps)
                Vn_fb = np.asarray(rec["Vn"], dtype=np.float32)
                if Vn_fb.ndim == 1:
                    Vn_fb = Vn_fb[None, :]
                Vn_fb = Vn_fb[:min(K_noise, Vn_fb.shape[0]), :]
                Vn_fb = Vn_fb / (np.linalg.norm(Vn_fb, axis=1, keepdims=True) + eps)

                agg[c]["Vr"] = vr_fb.astype(np.float32)
                agg[c]["Vn"] = Vn_fb.astype(np.float32)
                agg[c]["fallback_used"] = True
                agg[c]["best_cid"] = int(cid_fb)

    return agg