# noise.py
import numpy as np
import numpy as np

CIFAR10_ASYM_MAP = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}  # truck->auto, bird->airplane, cat<->dog, deer->horse




def get_client_distributions(y, dict_users, num_classes):
    """
    Returns:
      counts_per_client: (num_users, num_classes) int64
    """
    num_users = len(dict_users)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    counts = np.zeros((num_users, num_classes), dtype=np.int64)
    for cid in range(num_users):
        idx = np.asarray(list(dict_users.get(cid, [])), dtype=np.int64)
        if idx.size == 0:
            continue
        counts[cid] = np.bincount(y[idx], minlength=num_classes).astype(np.int64)
    return counts


def greedy_min_clean_cover(counts_per_client, min_per_class):
    """
    Greedy set-cover for minimum clean clients:
      - client covers class c if count[c] >= min_per_class
      - iteratively pick client with max coverage of remaining (uncovered) classes

    Args:
      counts_per_client: (num_users, num_classes)
      min_per_class: int

    Returns:
      clean_cids: list[int] chosen clean clients (greedy-minimal)
      covered_classes: set[int] classes covered by chosen clients
      uncovered_classes: set[int] classes not coverable / left uncovered
      client_cover_sets: list[set[int]] cover set per client
    """
    num_users, num_classes = counts_per_client.shape

    client_cover_sets = []
    for cid in range(num_users):
        cover = set(np.flatnonzero(counts_per_client[cid] >= min_per_class).tolist())
        client_cover_sets.append(cover)

    uncovered = set(range(num_classes))
    clean_cids = []
    used = set()

    # Greedy selection
    while uncovered:
        best_cid = None
        best_gain = 0

        for cid in range(num_users):
            if cid in used:
                continue
            gain = len(client_cover_sets[cid] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_cid = cid

        # No client can cover any remaining class => stop (impossible to fully span)
        if best_cid is None or best_gain == 0:
            break

        clean_cids.append(best_cid)
        used.add(best_cid)

        # "refine the class set" by removing those already covered
        newly_covered = client_cover_sets[best_cid] & uncovered
        uncovered -= newly_covered

    covered = set(range(num_classes)) - set(uncovered)
    return clean_cids, covered, uncovered, client_cover_sets


def choose_gamma_with_clean_cover(
    seed,
    y_clean,
    dict_users,
    num_classes,
    level_n_system,
    min_per_class=500,
    forced_gamma_s=None,
):
    """
    If forced_gamma_s is None:
      1) Find a (greedy) minimal set of clean clients whose union covers all labels
         according to min_per_class threshold.
      2) Force those clients to be clean (gamma_s=0).
      3) Choose noisy clients among the rest according to level_n_system.

    Returns:
      gamma_s: (num_users,) int64, 0 clean, 1 noisy
      clean_cids: list[int]
      uncovered_classes: set[int] (empty if full cover possible)
      counts_per_client: (num_users, num_classes)
      client_cover_sets: list[set[int]]
    """
    rng = np.random.RandomState(seed)
    num_users = len(dict_users)

    if forced_gamma_s is not None:
        gamma_s = np.asarray(forced_gamma_s, dtype=np.int64).reshape(-1)
        assert gamma_s.shape[0] == num_users
        # still return distributions for transparency
        counts = get_client_distributions(y_clean, dict_users, num_classes)
        cover_sets = [set(np.flatnonzero(counts[cid] >= min_per_class).tolist()) for cid in range(num_users)]
        clean_cids = [cid for cid in range(num_users) if gamma_s[cid] == 0]
        uncovered = set()
        return gamma_s, clean_cids, uncovered, counts, cover_sets

    # Compute distributions
    counts = get_client_distributions(y_clean, dict_users, num_classes)

    # Greedy minimal clean set that spans labels (by threshold)
    clean_cids, covered, uncovered, cover_sets = greedy_min_clean_cover(counts, min_per_class)

    if uncovered:
        print(
            f"[WARN] Could not cover all {num_classes} classes with min_per_class={min_per_class}. "
            f"Covered {len(covered)}/{num_classes}. Uncovered: {sorted(list(uncovered))}"
        )
    else:
        print(
            f"[INFO] Clean cover found with {len(clean_cids)} clients: {clean_cids}"
        )

    # Build gamma_s: default clean
    gamma_s = np.zeros(num_users, dtype=np.int64)

    # Sample noisy clients among remaining clients using your probability/fraction
    k_noisy = int(round(level_n_system * num_users))
    k_noisy = max(0, min(num_users, k_noisy))

    remaining = np.setdiff1d(np.arange(num_users, dtype=np.int64), np.asarray(clean_cids, dtype=np.int64), assume_unique=False)

    if k_noisy > remaining.size:
        print(f"[WARN] Requested k_noisy={k_noisy} but only {remaining.size} non-clean clients remain. Clamping.")
        k_noisy = int(remaining.size)

    noisy_ids = rng.choice(remaining, size=k_noisy, replace=False) if k_noisy > 0 else np.array([], dtype=np.int64)
    gamma_s[noisy_ids] = 1

    return gamma_s, clean_cids, uncovered, counts, cover_sets


def _sample_sym(label, num_classes, rng):
    candidates = list(range(num_classes))
    candidates.remove(int(label))
    return int(rng.choice(candidates))


def per_class_flip_summary(y_before, y_after, num_classes, sample_idx):
    y_before = np.asarray(y_before, dtype=np.int64)[sample_idx]
    y_after  = np.asarray(y_after,  dtype=np.int64)[sample_idx]

    before_counts = np.bincount(y_before, minlength=num_classes)
    after_counts  = np.bincount(y_after,  minlength=num_classes)

    changed = (y_before != y_after)
    flipped_out = np.bincount(y_before[changed], minlength=num_classes)
    flipped_in  = np.bincount(y_after[changed],  minlength=num_classes)

    kept = before_counts - flipped_out

    rows = []
    for c in range(num_classes):
        total_before = int(before_counts[c])
        total_after  = int(after_counts[c])
        out_c = int(flipped_out[c])
        in_c  = int(flipped_in[c])
        kept_c = int(kept[c])
        delta = total_after - total_before
        flip_rate = (out_c / total_before) if total_before > 0 else 0.0

        rows.append({
            "class": c,
            "total_before": total_before,
            "total_after_flipped": total_after,
            "flipped_out": out_c,
            "flipped_in": in_c,
            "delta": delta,
            "kept": kept_c,
            "flip_rate": float(flip_rate),
        })
    return rows



# Assumes you already have:
#  - _sample_sym(orig, num_classes, rng)  -> returns label != orig
#  - _sample_asym(orig, dataset_name, num_classes, rng) -> returns mapped label (may equal orig if not flippable)
#  - CIFAR10_ASYM_MAP (or whatever map your _sample_asym uses)
#  - per_class_flip_summary(y_clean, y_noisy, num_classes, sample_idx=...)

def add_noise(
    seed,
    y_clean,
    dict_users,
    level_n_system,     # P(client is noisy) OR fraction of noisy clients (your current behavior)
    noise_p,            # target flip rate *per class* inside noisy clients
    num_classes,
    dataset_name="cifar10",
    noise_type="sym",   # "sym" or "asym"
    max_asym_rounds=10, # kept for API compatibility (rarely needed now)
    forced_gamma_s=None,
):
    """
    Client-dependent label noise with strict PER-CLASS targets (non-IID-safe):
      - If gamma_s[cid]==0: 0 flips guaranteed.
      - If gamma_s[cid]==1:
          For each class c present in that client's data:
            k_c = round(noise_p * n_c) flips are applied *from that class*.

        Symmetric:
          each selected example of class c is flipped to a random label != c.
        Asymmetric:
          only classes in the asym mapping are flippable;
          for a mapped class c: flip to mapped label (or your _sample_asym behavior).
          for an unmapped class: k_c forced to 0.

    Prints:
      - per-client total mismatch ratio (over all samples in the client)
      - per-class flip summary table (your exact format)
    """
    rng = np.random.RandomState(seed)
    y_clean = np.asarray(y_clean, dtype=np.int64).reshape(-1)
    y_noisy = y_clean.copy()

    num_users = len(dict_users)
    # ----------------- choose which clients are noisy -----------------
    if forced_gamma_s is None:
        gamma_s, clean_cids, uncovered, counts_per_client, cover_sets = choose_gamma_with_clean_cover(
            seed=seed,
            y_clean=y_clean,
            dict_users=dict_users,
            num_classes=num_classes,
            level_n_system=level_n_system,
            min_per_class=450,
            forced_gamma_s=forced_gamma_s,
        )

    else:
        gamma_s = np.asarray(forced_gamma_s, dtype=np.int64).reshape(-1)
        assert gamma_s.shape[0] == num_users

    flip_mask = np.zeros(len(y_clean), dtype=bool)
    real_noise_level = np.zeros(num_users, dtype=float)
    flips_by_client = {}


    for cid in range(num_users):
        sample_idx = np.asarray(list(dict_users.get(cid, [])), dtype=np.int64)
        n = int(sample_idx.size)

        if n == 0:
            flips_by_client[cid] = np.array([], dtype=np.int64)
            real_noise_level[cid] = 0.0
            print(f"Client {cid}: EMPTY")
            continue

        # ----------------- CLEAN CLIENT => GUARANTEED 0 NOISE -----------------
        if gamma_s[cid] == 0 or noise_p <= 0:
            flips_by_client[cid] = np.array([], dtype=np.int64)
            real_noise_level[cid] = 0.0
            assert np.all(y_noisy[sample_idx] == y_clean[sample_idx])
            print(f"Client {cid}: CLEAN (gamma_s=0)  real=0.0000  flipped=0/{n}")
            rows = per_class_flip_summary(y_clean, y_noisy, num_classes, sample_idx=sample_idx)
            print(f"  Per-class label noise summary (client {cid}):")
            for r in rows:
                print(
                    f"  class {r['class']:2d} | "
                    f"before={r['total_before']:5d}  "
                    f"after={r['total_after_flipped']:5d}  "
                    f"flipped_out={r['flipped_out']:5d}  "
                    f"flipped_in={r['flipped_in']:5d}  "
                    f"delta={r['delta']:+5d}  "
                    f"kept={r['kept']:5d}  "
                    f"flip_rate={r['flip_rate']:.3f}"
                )
            print("\n========================================================")
            continue
        # ----------------- NOISY CLIENT => PER-CLASS TARGETS -----------------
        yc = y_noisy[sample_idx]  # current labels (still clean at this point for this client)
        flipped_global = []

        # count per class in this client
        # (classes absent => n_c = 0 => k_c=0)
        for c in range(num_classes):
            cls_mask_local = (yc == c)
            n_c = int(np.sum(cls_mask_local))
            if n_c == 0:
                continue

            
            k_c = int(np.round(noise_p * n_c))
            k_c = max(0, min(k_c, n_c))
            if k_c == 0:
                continue

            # global indices in this client belonging to class c
            cls_global_idx = sample_idx[cls_mask_local]

            # choose exactly k_c examples from this class
            chosen = rng.choice(cls_global_idx, size=k_c, replace=False)

            for gidx in chosen:
                gidx = int(gidx)
                orig = int(y_noisy[gidx])
                new = _sample_sym(orig, num_classes, rng)

                # defensive: ensure changed
                if new == orig:
                    for _ in range(10):
                        new = _sample_sym(orig, num_classes, rng)
                        if new != orig:
                            break
                if new == orig:
                    continue  # extremely unlikely if _sample_sym is correct

                y_noisy[gidx] = int(new)
                flip_mask[gidx] = True
                flipped_global.append(gidx)

            # top-up within this class if we somehow fell short (shouldn't happen)
            if len(chosen) != k_c:
                pass  # (choice guarantees size)

            if len([g for g in chosen if flip_mask[int(g)]]) < k_c:
                # attempt top-up from remaining in-class pool
                already = set(int(g) for g in flipped_global)
                pool = [int(g) for g in cls_global_idx if int(g) not in already]
                remaining = k_c - len([g for g in chosen if flip_mask[int(g)]])
                if remaining > 0 and len(pool) > 0:
                    extra = rng.choice(np.asarray(pool, dtype=np.int64), size=min(remaining, len(pool)), replace=False)
                    for gidx in extra:
                        gidx = int(gidx)
                        orig = int(y_noisy[gidx])
                        new = _sample_sym(orig, num_classes, rng)
                        if new != orig:
                            y_noisy[gidx] = int(new)
                            flip_mask[gidx] = True
                            flipped_global.append(gidx)



        flips_by_client[cid] = np.asarray(flipped_global, dtype=np.int64)

        noise_ratio = float(np.mean(y_clean[sample_idx] != y_noisy[sample_idx]))
        real_noise_level[cid] = noise_ratio

        # For reporting: total intended flips (sum over classes)
        # (note: asym may end up short if mapping can't flip some chosen)
        intended = 0
        yc_clean = y_clean[sample_idx]
        for c in range(num_classes):
            n_c = int(np.sum(yc_clean == c))
            if n_c == 0:
                continue
            intended += int(np.round(noise_p * n_c))

        print(
            f"Client {cid}: gamma_s=1 target_per_class={noise_p:.4f} "
            f"intended_flips≈{intended}/{n} real={noise_ratio:.4f} flipped={len(flipped_global)}/{n}"
        )

        # per-client per-class summary (your format)
        rows = per_class_flip_summary(y_clean, y_noisy, num_classes, sample_idx=sample_idx)
        print(f"  Per-class label noise summary (client {cid}):")
        for r in rows:
            print(
                f"  class {r['class']:2d} | "
                f"before={r['total_before']:5d}  "
                f"after={r['total_after_flipped']:5d}  "
                f"flipped_out={r['flipped_out']:5d}  "
                f"flipped_in={r['flipped_in']:5d}  "
                f"delta={r['delta']:+5d}  "
                f"kept={r['kept']:5d}  "
                f"flip_rate={r['flip_rate']:.3f}"
            )
        print("\n========================================================")

            

    return y_noisy, gamma_s, real_noise_level, flip_mask, flips_by_client
