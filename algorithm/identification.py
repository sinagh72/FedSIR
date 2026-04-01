from math import tau

import numpy as np
import os
from datetime import datetime
from model.build_model import build_backbone_model
from model.trainer import train_model
from utils.metrics import  client_identification_metrics
from algorithm.fit_gmm import multivariate_gmm_clean_noisy
from algorithm.helper import state_dict_to_cpu
from algorithm.eigen import compute_feats_eigen, extract_backbone_features
from utils.similarity_plot import  plot_matrix
import numpy as np

def sim_matrix_diagnostics(M, eps=1e-12, tau=0.7):
    """
    M: [K,K] similarity matrix with diag ~ 1, off-diag in [0,1], may contain NaNs.
    Returns:
      stats: dict of useful diagnostics
    """
    M = np.asarray(M, dtype=np.float64)
    K = M.shape[0]
    assert M.shape[1] == K

    stats = {"K": K}

    # full-size masks
    diag_mask = np.eye(K, dtype=bool)
    off_mask = ~diag_mask
    finite_mask = np.isfinite(M)
    finite_off_mask = finite_mask & off_mask

    # ---------------- 0) coverage ----------------
    off_vals = M[finite_off_mask]
    if off_vals.size == 0:
        stats.update({
            "off_mean": 0.0,
            "off_std": 0.0,
            "off_max": 0.0,
            "off_p90": 0.0,
            "off_p95": 0.0,
            "off_energy": 0.0,
            "row_off_mean_mean": 0.0,
            "row_max_mean": 0.0,
            "row_top2_gap_mean": 0.0,
            "row_entropy_mean": 0.0,
            "lambda1": 0.0,
            "lambda1_ratio": 0.0,
            "effective_rank": 0.0,
            "edge_density_tau": 0.0,
        })
        return stats

    row_valid_counts = np.sum(finite_off_mask, axis=1)
    usable_classes = np.where(row_valid_counts >= 1)[0]   # at least one valid off-diagonal
    # ---------------- 1) global off-diagonal stats ----------------
    stats["off_mean"] = float(off_vals.mean())
    stats["off_std"] = float(off_vals.std(ddof=0))
    stats["off_max"] = float(off_vals.max())
    stats["off_p90"] = float(np.quantile(off_vals, 0.90))
    stats["off_p95"] = float(np.quantile(off_vals, 0.95))
    stats["off_energy"] = float(np.mean(off_vals ** 2))

    # ---------------- 2) row peakedness + entropy ----------------
    row_max = []
    row_gap = []
    row_ent = []
    row_mean = []
    row_max_minus_mean = []

    for i in range(K):
        row = M[i, :]
        vals = row[finite_off_mask[i, :]]   # only finite off-diagonal entries

        if vals.size >= 1:
            row_mean.append(float(np.mean(vals)))
            row_max.append(float(np.max(vals)))

        if vals.size >= 2:
            s = np.sort(vals)
            mx = s[-1]
            sc = s[-2]
            row_gap.append(float(mx - sc))
            row_max_minus_mean.append(float(mx - np.mean(vals)))
            # normalized entropy for comparability across different row lengths
            w = vals - np.min(vals)
            w = w + eps
            p = w / (w.sum() + eps)
            ent = float(-(p * np.log(p + eps)).sum())
            ent_norm = ent / np.log(len(vals) + eps)
            row_ent.append(ent_norm)
    stats["row_off_mean"] = float(np.mean(row_mean)) if row_mean else 0.0
    stats["row_max_mean"] = float(np.mean(row_max)) if row_max else 0.0
    stats["row_top2_gap_mean"] = float(np.mean(row_gap)) if row_gap else 0.0
    stats["row_max_minus_mean"] = float(np.mean(row_max_minus_mean)) if row_max_minus_mean else 0.0
    stats["row_entropy_mean"] = float(np.mean(row_ent)) if row_ent else 0.0

    # ---------------- 3) spectral shape on usable induced submatrix ----------------
    # use classes that have at least one valid off-diagonal
    usable = usable_classes
    if len(usable) < 2:
        stats["lambda1"] = 0.0
        stats["lambda1_ratio"] = 0.0
        stats["effective_rank"] = 0.0
        stats["edge_density_tau"] = 0.0
        return stats

    A = M[np.ix_(usable, usable)].copy()

    # sanity check: this submatrix should now be finite off-diagonal
    # if not, something upstream is inconsistent
    if not np.all(np.isfinite(A[~np.eye(A.shape[0], dtype=bool)])):
        stats["lambda1"] = 0.0
        stats["lambda1_ratio"] = 0.0
        stats["effective_rank"] = 0.0
        stats["edge_density_tau"] = 0.0
        return stats

    np.fill_diagonal(A, 0.0)
    A = 0.5 * (A + A.T)

    evals = np.linalg.eigvalsh(A)
    evals = np.clip(evals, 0.0, None)
    sume = float(evals.sum())

    if sume <= eps:
        stats["lambda1"] = 0.0
        stats["lambda1_ratio"] = 0.0
        stats["effective_rank"] = 0.0
    else:
        lam1 = float(evals[-1])
        stats["lambda1"] = lam1
        stats["lambda1_ratio"] = lam1 / sume
        p = evals / (sume + eps)
        p = p[p > 0]
        H = float(-(p * np.log(p + eps)).sum())
        stats["effective_rank"] = float(np.exp(H))

    # ---------------- 4) threshold graph density ----------------
    K_sub = A.shape[0]
    off_mask_sub = ~np.eye(K_sub, dtype=bool)
    off_sub = A[off_mask_sub]
    edges = np.sum(off_sub > tau)
    stats["edge_density_tau"] = float(edges) / float(off_sub.size + eps)

    return stats


def simmat_to_features(M: np.ndarray, *, fill_value: float = 0.0):
    K = M.shape[0]
    iu = np.triu_indices(K, k=1)
    v = M[iu]

    finite = np.isfinite(v)
    frac_present = float(np.mean(finite))
    v = np.where(finite, v, fill_value).astype(np.float32)

    # add low-dim summaries that are very informative
    stats = np.array([
        frac_present,
        float(v.mean()),
        float(v.std()),
        float(np.quantile(v, 0.9)),
        float(np.quantile(v, 0.1)),
    ], dtype=np.float32)

    return np.concatenate([v, stats], axis=0)


def identification_process(cfg, num_classes, gamma_s, train_loaders_train, cfg_identification, base_state, save_dir, device):    
    eigs_by_client = {}
    similarity_intra_matrix = {}

    X_list = []
    feat_names = None
    client_states = []
    client_weights = []
    print(f"\n==================  Identification Process ==================")
    # ===================================== clean client identification =================================================
    for cid in range(cfg.num_users):
        net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
        net.load_state_dict(base_state, strict=True)  # same init for fairness
        print(f"\n--- Client {cid} Identification Training ---")
        net, _ = train_model(net, train_loaders_train[cid], device, cfg_identification, cfg, label_mode="noisy")
        client_states.append(state_dict_to_cpu(net.state_dict()))
        client_weights.append(len(train_loaders_train[cid].dataset))

        client_features = extract_backbone_features(net, train_loaders_train[cid], device)
        class_counts = {c: rec["feats"].shape[0] for c, rec in client_features.items()}

        min_count = 20 
        maj_classes = sorted([c for c, n in class_counts.items() if n >= min_count])
        # fallback: ensure we can form at least a 2x2 off-diagonal set
        if len(maj_classes) < 2:
            # too non-IID / too small; mark as suspicious or skip from GMM
            maj_classes = sorted(class_counts.keys())  # or []

        eigs_by_client[cid] = compute_feats_eigen(client_features, allowed_classes=maj_classes)

        M = np.full((num_classes, num_classes), np.nan, dtype=np.float32)
        np.fill_diagonal(M, 1.0)

        keys = sorted(eigs_by_client[cid].keys())
        for i in keys:
            v_s_i = eigs_by_client[cid][i]["Vr"]
            for j in keys:
                if i == j:
                    continue
                v_s_j = eigs_by_client[cid][j]["Vr"]
                M[i, j] = np.abs(v_s_i @ v_s_j) 
        similarity_intra_matrix[cid] = M
        
        plot_matrix(input_matrix=similarity_intra_matrix[cid], num_classes=10, save_dir=save_dir, plot_name=f"sim_matrix_{cid}")
        stats = sim_matrix_diagnostics(M, tau=0.7)
        # print(f"[cid {cid}] sim-matrix stats:", stats)
        print(f"[cid {cid}] stats:", stats)
        
        x2 = np.array([
            stats["off_mean"],
            stats["off_energy"],
        ])
        x = x2
        X_list.append(x)
        print(x)

    X = np.stack(X_list, axis=0)  # [N_clients, D_features]
    res = multivariate_gmm_clean_noisy(X, seed=cfg.seed)
    metrics = client_identification_metrics(gamma_s, res, num_users=cfg.num_users)

    print(f"Identification accuracy: {metrics['acc']*100:.2f}%")
    print(f"TP(clean)={metrics['TP']} TN(noisy)={metrics['TN']} "
        f"FP(noisy→clean)={metrics['FP']} FN(clean→noisy)={metrics['FN']}")

    print("Wrong client IDs:", metrics["wrong_clients"])
    print("Clean clients: ", res["clean_idx"].tolist())
    print("Noisy clients: ", res["noisy_idx"].tolist())

    log_path = os.path.join(save_dir, f"{gamma_s.tolist().count(0)}_relabeling.txt")
    os.makedirs(save_dir, exist_ok=True)

    with open(log_path, "a") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Accuracy: {metrics['acc']*100:.2f}%\n")

        # clean is positive
        f.write(f"TP(clean)={metrics['TP']} TN(noisy)={metrics['TN']} "
                f"FP(noisy->clean)={metrics['FP']} FN(clean->noisy)={metrics['FN']}\n")

        f.write(f"Clean recall:    {metrics['clean_recall']*100:.2f}%\n")
        f.write(f"Clean precision: {metrics['clean_precision']*100:.2f}%\n")
        f.write(f"Clean F1:        {metrics['clean_f1']*100:.2f}%\n")
        f.write(f"Noisy recall:    {metrics['noisy_recall']*100:.2f}%\n")

        f.write(f"Wrong client IDs: {metrics['wrong_clients']}\n")
        f.write(f"Gamma_s (0=clean,1=noisy): {gamma_s}\n")

        f.write(f"Component clean-scores (higher=cleaner): {res['comp_clean_score']}\n")
        f.write(f"Clean component: {res['clean_comp']}  Noisy component: {res['noisy_comp']}\n")
        f.write(f"Clean clients: {res['clean_idx'].tolist()}\n")
        f.write(f"Noisy clients: {res['noisy_idx'].tolist()}\n")

        f.write(f"Noise type: {cfg.noise_type}\n")
        f.write(f"Noise level: {cfg.noise_p}\n")

    
    rank_noisy = np.argsort(res["p_clean"])   # smallest p_clean = most noisy
    print("Most suspicious (top 10):", rank_noisy.tolist())
    
    # ---- 
    clean_cids = res["clean_idx"].tolist()
    noisy_cids = res["noisy_idx"].tolist()

    if len(clean_cids) == 0:
        print("\n[WARN] No clean clients found. Training ALL clients as noisy (no relabeling).")

    
    return client_states, client_weights, clean_cids, noisy_cids