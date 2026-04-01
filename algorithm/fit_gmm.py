import numpy as np
from sklearn.mixture import GaussianMixture

def multivariate_gmm_clean_noisy(
    X: np.ndarray,
    seed: int = 0,
    *,
    clean_feature_idx: int = 0,     # index of agree_rate in your x vector
    covariance_type: str = "full",  # "full" is safer than "diag" for correlated stats
    prob_threshold: float = 0.5,
):
    """
    Fit 2-component GMM on client features and return a clean/noisy split.

    Key fix: determine which component is 'clean' by looking at a feature that
    correlates with cleanliness. For your current feature vector:
      x = [agree_rate, sr_margin_med, sn_margin_med]
    we set clean_feature_idx=0 (agree_rate), and clean component = higher mean agree_rate.

    Returns:
      p_clean: [N] probability of being in clean component
      clean_idx / noisy_idx
      clean_comp / noisy_comp
      gmm (fitted model)
      comp_clean_score: diagnostic scalar per component (higher => cleaner)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D [N,D], got shape {X.shape}")
    N, D = X.shape
    if N < 2:
        raise ValueError("Need >=2 clients.")

    # ---- handle NaN/inf robustly ----
    # If you ever pass NaNs (e.g., from masked features), replace per-feature with column mean
    X2 = X.copy()
    bad = ~np.isfinite(X2)
    if bad.any():
        col_mean = np.nanmean(np.where(np.isfinite(X2), X2, np.nan), axis=0)
        # if a whole column is NaN, set it to 0
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        X2[bad] = np.take(col_mean, np.where(bad)[1])

    # degenerate: all clients identical
    if np.allclose(X2, X2[0]):
        clean_mask = np.ones(N, dtype=bool)
        return {
            "p_clean": np.ones(N),
            "clean_idx": np.arange(N),
            "noisy_idx": np.array([], dtype=int),
            "clean_comp": 0,
            "noisy_comp": 1,
            "comp_clean_score": np.array([0.0, 0.0], dtype=np.float64),
            "gmm": None,
        }

    # ---- standardize ----
    mu = X2.mean(axis=0, keepdims=True)
    sig = X2.std(axis=0, keepdims=True) + 1e-12
    Z = (X2 - mu) / sig

    # ---- fit GMM ----
    gmm = GaussianMixture(
        n_components=2,
        covariance_type=covariance_type,  # "full" generally more stable than "diag" here
        reg_covar=1e-6,
        n_init=30,
        max_iter=2000,
        random_state=seed,
    ).fit(Z)

    means2 = gmm.means_  # [2, D] in standardized space

    # ---- choose clean component by a monotonic cleanliness feature ----
    if not (0 <= clean_feature_idx < D):
        raise ValueError(f"clean_feature_idx={clean_feature_idx} out of bounds for D={D}")

    # In your current setup, feature 0 = agree_rate (higher => cleaner)
    comp_clean_score = means2[:, clean_feature_idx].astype(np.float64)  # higher => cleaner

    clean_comp = int(np.argmin(comp_clean_score))
    noisy_comp = 1 - clean_comp

    # ---- posterior probability of clean ----
    p_clean = gmm.predict_proba(Z)[:, clean_comp]
    clean_mask = p_clean >= float(prob_threshold)

    return {
        "p_clean": p_clean,
        "clean_idx": np.where(clean_mask)[0],
        "noisy_idx": np.where(~clean_mask)[0],
        "clean_comp": clean_comp,
        "noisy_comp": noisy_comp,
        "comp_clean_score": comp_clean_score,
        "gmm": gmm,
        "mu": mu.reshape(-1),
        "sig": sig.reshape(-1),
    }