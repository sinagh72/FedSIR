import numpy as np
import os


def client_identification_metrics(gamma_s, res, num_users=None, threshold=0.5):
    """
    Clean is the POSITIVE class.

    gamma_s: [U], 0=clean, 1=noisy (ground truth)
    res: dict from multivariate_gmm_clean_noisy()
         must contain either:
           - clean_idx & noisy_idx
           OR
           - p_clean (probability of clean)

    Returns:
        TP = correctly predicted clean
        TN = correctly predicted noisy
        FP = predicted clean but actually noisy
        FN = predicted noisy but actually clean
    """

    y_true_noise = np.asarray(gamma_s, dtype=np.int64)
    U = int(num_users) if num_users is not None else y_true_noise.shape[0]

    # ---- predicted in noise convention (0=clean, 1=noisy) ----
    if "clean_idx" in res and "noisy_idx" in res:
        y_pred_noise = np.zeros(U, dtype=np.int64)
        y_pred_noise[np.asarray(res["noisy_idx"], dtype=np.int64)] = 1
    elif "p_clean" in res:
        p_clean = np.asarray(res["p_clean"], dtype=float)
        y_pred_noise = (p_clean < threshold).astype(np.int64)  # 1=noisy
    else:
        raise ValueError("res must contain (clean_idx & noisy_idx) or p_clean")

    # ---- convert to clean-positive convention ----
    y_true = (y_true_noise == 0).astype(np.int64)  # 1=clean, 0=noisy
    y_pred = (y_pred_noise == 0).astype(np.int64)  # 1=clean, 0=noisy

    # ---- confusion matrix ----
    TP = int(((y_true == 1) & (y_pred == 1)).sum())  # clean correctly predicted clean
    TN = int(((y_true == 0) & (y_pred == 0)).sum())  # noisy correctly predicted noisy
    FP = int(((y_true == 0) & (y_pred == 1)).sum())  # noisy predicted clean
    FN = int(((y_true == 1) & (y_pred == 0)).sum())  # clean predicted noisy

    acc = (TP + TN) / max(1, U)

    # Clean metrics (positive class)
    clean_recall = TP / max(1, (y_true == 1).sum())
    clean_precision = TP / max(1, (TP + FP))
    clean_f1 = 2 * clean_precision * clean_recall / max(
        1e-12, (clean_precision + clean_recall)
    )

    # Noisy recall (how many noisy correctly caught)
    noisy_recall = TN / max(1, (y_true == 0).sum())

    wrong = np.where(y_true != y_pred)[0].tolist()

    return {
        "acc": acc,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "clean_recall": clean_recall,
        "clean_precision": clean_precision,
        "clean_f1": clean_f1,
        "noisy_recall": noisy_recall,
        "wrong_clients": wrong,
        "y_pred_clean": y_pred,      # 1=clean
        "y_pred_noise": y_pred_noise # 1=noisy (original convention)
    }


def bootstrap_ci(
    y_true,
    y_pred,
    per_sample_losses,
    num_classes,
    B=1000,
    seed=123,
    alpha=0.05,  # 0.05 => 95% CI
):
    """
    Returns dict:
      metric -> (mean, delta_low, delta_high)
    where:
      delta_low  = mean - CI_low
      delta_high = CI_high - mean
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    per_sample_losses = np.asarray(per_sample_losses, dtype=float).reshape(-1)

    N = y_true.size
    rng = np.random.default_rng(seed)

    accs, macro_f1s, micro_f1s, losses = [], [], [], []

    for _ in range(B):
        idx = rng.integers(0, N, size=N)  # sample WITH replacement
        m = per_class_metrics(y_true[idx], y_pred[idx], num_classes=num_classes)
        accs.append(float(m["acc"]))
        macro_f1s.append(float(m["macro"]["f1"]))
        micro_f1s.append(float(m["micro"]["f1"]))
        losses.append(float(per_sample_losses[idx].mean()))

    def mean_deltas(v):
        v = np.asarray(v, dtype=float)
        mean = float(v.mean())
        ci_low = float(np.quantile(v, alpha / 2))
        ci_high = float(np.quantile(v, 1 - alpha / 2))
        dlow = mean - ci_low
        dhigh = ci_high - mean
        return mean, dlow, dhigh

    return {
        "acc": mean_deltas(accs),
        "macro_f1": mean_deltas(macro_f1s),
        "micro_f1": mean_deltas(micro_f1s),
        "loss": mean_deltas(losses),
    }


def per_class_metrics(y_true, y_pred, num_classes, eps=1e-12):
    """
    Multi-class, per-class metrics using one-vs-rest:
      TP_c = #(y_true=c and y_pred=c)
      FP_c = #(y_true!=c and y_pred=c)
      FN_c = #(y_true=c and y_pred!=c)
      TN_c = #(y_true!=c and y_pred!=c)

    Returns:
      metrics: dict with arrays of length C for TP/FP/FN/TN, precision, recall, f1, support
      plus macro/micro summaries.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    C = int(num_classes)
    conf = np.zeros((C, C), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < C) & (y_pred >= 0) & (y_pred < C)
    yt = y_true[valid]
    yp = y_pred[valid]

    # confusion matrix: rows=true, cols=pred
    np.add.at(conf, (yt, yp), 1)

    TP = np.diag(conf).astype(np.int64)
    FP = conf.sum(axis=0) - TP
    FN = conf.sum(axis=1) - TP
    TN = conf.sum() - (TP + FP + FN)

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    support   = conf.sum(axis=1)

    # overall accuracy
    acc = TP.sum() / (conf.sum() + eps)

    # macro averages
    macro_precision = precision.mean()
    macro_recall    = recall.mean()
    macro_f1        = f1.mean()

    # micro averages (for single-label multiclass, micro P=R=F1=accuracy)
    micro_TP = TP.sum()
    micro_FP = FP.sum()
    micro_FN = FN.sum()
    micro_precision = micro_TP / (micro_TP + micro_FP + eps)
    micro_recall    = micro_TP / (micro_TP + micro_FN + eps)
    micro_f1        = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)

    return {
        "confusion": conf,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision, "recall": recall, "f1": f1, "support": support,
        "acc": acc,
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        
    }


def save_metrics_report_txt_with_ci(
    path,
    title,
    metrics_dict,     # output of per_class_metrics 
    num_classes,
    ci_deltas=None,   # output of bootstrap_ci_deltas_only (can be None)
    class_names=None,
    extra_lines=None,
):
    """
    Same as save_metrics_report_txt, but overall metric lines include:
      mean ± avg_half_width (+Δ_high / -Δ_low)
    using ci_deltas[metric] = (mean, dlow, dhigh).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def fmt_overall(key, label, value):
        # value is your point estimate from full test set
        if ci_deltas is None or key not in ci_deltas:
            return f"{label}:  {value:.6f}\n"

        mean_b, dlow, dhigh = ci_deltas[key]

        # We display uncertainty from bootstrap (mean_b, CI) but keep the point estimate value.
        # If prefer using mean_b as the displayed center, replace `value` with `mean_b`.
        half = 0.5 * (dlow + dhigh)  # average half-width for the ± part

        return (f"{label}:  {value:.6f}  ± {half:.6f}  "
                f"(+{dhigh:.6f} / -{dlow:.6f})\n")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(title + "\n")
        f.write("=" * 80 + "\n\n")

        if extra_lines:
            for line in extra_lines:
                f.write(str(line) + "\n")
            f.write("-" * 80 + "\n")

        # ---- overall ----
        f.write(fmt_overall("acc", "Overall acc", metrics_dict["acc"]))
        f.write(fmt_overall("macro_f1", "Macro  F1", metrics_dict["macro"]["f1"]))
        f.write(fmt_overall("micro_f1", "Micro  F1", metrics_dict["micro"]["f1"]))
        if "loss" in metrics_dict:
            f.write(fmt_overall("loss", "Loss", metrics_dict["loss"]))

        # ---- per-class (unchanged) ----
        f.write("\nPer-class:\n")
        f.write("class\tname\tprecision\trecall\tf1\tsupport\tTP\tFP\tFN\tTN\n")

        for c in range(num_classes):
            name = class_names[c] if class_names is not None else str(c)
            f.write(
                f"{c}\t{name}\t"
                f"{metrics_dict['precision'][c]:.6f}\t{metrics_dict['recall'][c]:.6f}\t{metrics_dict['f1'][c]:.6f}\t"
                f"{metrics_dict['support'][c]}\t{metrics_dict['TP'][c]}\t{metrics_dict['FP'][c]}\t"
                f"{metrics_dict['FN'][c]}\t{metrics_dict['TN'][c]}\n"
            )


def evaluate_relabeling(
    *,
    y_clean: np.ndarray,
    y_noisy: np.ndarray,      # labels before this stage
    y_after: np.ndarray,      # labels after this stage
    relabeled_mask: np.ndarray,  # True where THIS stage changed label
    stage_name: str = "",
):
    y_clean = np.asarray(y_clean).reshape(-1)
    y_noisy = np.asarray(y_noisy).reshape(-1)
    y_after = np.asarray(y_after).reshape(-1)
    relabeled_mask = np.asarray(relabeled_mask).reshape(-1).astype(bool)

    N = y_clean.size
    assert y_noisy.size == N and y_after.size == N and relabeled_mask.size == N

    noisy_mask = (y_noisy != y_clean)  # ground-truth: was label wrong before stage?

    # ---- detection confusion (positive = "we relabeled") ----
    TP = int((relabeled_mask & noisy_mask).sum())
    FP = int((relabeled_mask & ~noisy_mask).sum())
    FN = int((~relabeled_mask & noisy_mask).sum())
    TN = int((~relabeled_mask & ~noisy_mask).sum())

    # ---- outcome / fix quality ----
    TP_fixed   = int((relabeled_mask & noisy_mask & (y_after == y_clean)).sum())
    failed_fix = int((relabeled_mask & noisy_mask & (y_after != y_clean)).sum())

    # FP can be split too:
    FP_became_wrong = int((relabeled_mask & ~noisy_mask & (y_after != y_clean)).sum())
    FP_still_correct = int((relabeled_mask & ~noisy_mask & (y_after == y_clean)).sum())  # rare but possible

    bad_relabel = int((relabeled_mask & (y_after != y_clean)).sum())  # = failed_fix + FP_became_wrong

    # ---- rates ----
    noise_before = float(np.mean(noisy_mask))
    noise_after  = float(np.mean(y_after != y_clean))
    relabeled_count = int(relabeled_mask.sum())

    precision_detect = (TP / (TP + FP)) if (TP + FP) > 0 else float("nan")
    recall_detect    = (TP / (TP + FN)) if (TP + FN) > 0 else float("nan")

    print(f"\n=== {stage_name} ===")
    print(f"Relabeled: {relabeled_count} / {N}")
    print(f"Noise vs clean: before={noise_before:.4f}  after={noise_after:.4f}")

    print(f"Detection TP/TN/FP/FN: {TP} / {TN} / {FP} / {FN}")
    print(f"Detection precision: {precision_detect:.4f}   recall: {recall_detect:.4f}")

    print(f"Correctly fixed (TP_fixed): {TP_fixed}")
    print(f"Failed fixes (noisy but still wrong): {failed_fix}")
    print(f"False relabel (was clean): {FP}  (became wrong: {FP_became_wrong}, stayed correct: {FP_still_correct})")
    print(f"Bad relabel total: {bad_relabel}")

    return {
        "N": N,
        "relabeled_count": relabeled_count,
        "noise_before": noise_before,
        "noise_after": noise_after,

        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision_detect": precision_detect,
        "recall_detect": recall_detect,

        "TP_fixed": TP_fixed,
        "failed_fix": failed_fix,
        "FP_became_wrong": FP_became_wrong,
        "FP_still_correct": FP_still_correct,
        "bad_relabel": bad_relabel,
    }

