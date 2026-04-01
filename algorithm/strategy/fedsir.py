import os
import copy
import numpy as np
import torch
import torch.nn as nn
from model.build_model import build_backbone_model
from model.trainer import predict
from utils.metrics import (
    bootstrap_ci,
    per_class_metrics,
    save_metrics_report_txt_with_ci,
    evaluate_relabeling,
)
from algorithm.helper import weighted_average_state_dicts, state_dict_to_cpu, daagg, iter_client_batches, iter_client_batches_with_idx, collect_client_class_counts
from algorithm.identification import identification_process
from algorithm.eigen import compute_feats_eigen, extract_backbone_features
from algorithm.relabel import relabel_noisy_samples_from_clean_dict
from model.loss import LogitAdjust, LA_KD, CE_KD
# ============================================================
# FedSIR utilities
# ============================================================
def sigmoid_rampup(current, begin, end):
    current = np.clip(current, begin, end)
    phase = 1.0 - (current - begin) / max(1e-12, (end - begin))
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(rnd, begin, end):
    return sigmoid_rampup(rnd, begin, end)
# ============================================================
# Diagnostics helpers 
# ============================================================
def compute_flip_rate_from_maps(prev_map, cur_map):
    if prev_map is None or len(prev_map) == 0:
        return 1.0, 0, max(1, len(cur_map))

    common_keys = prev_map.keys() & cur_map.keys()
    if len(common_keys) == 0:
        return 1.0, 0, 1

    n_changed = 0
    for k in common_keys:
        if int(prev_map[k]) != int(cur_map[k]):
            n_changed += 1

    n_total = len(common_keys)
    flip_rate = float(n_changed) / max(1, n_total)
    return flip_rate, n_changed, n_total
# ============================================================
# Aggregatopmn helper
# ============================================================
def aggregate_models(w_locals, dict_len, clean_clients, noisy_clients, mode="daagg"):
    if mode == "daagg":
        return daagg(
            w_locals=w_locals,
            dict_len=dict_len,
            clean_clients=clean_clients,
            noisy_clients=noisy_clients,
        )
    elif mode == "weighted_avg":
        return weighted_average_state_dicts(w_locals, dict_len)
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")
    
# ============================================================
# Local training
# ============================================================
def _local_train_clean(
    net,
    train_loader,
    device,
    lr,
    local_ep,
    cls_num_list,
    loss_mode="la",   # "la" or "ce"
):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)

    if loss_mode == "la":
        criterion = LogitAdjust(cls_num_list=cls_num_list, device=device)
    elif loss_mode == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown clean loss_mode: {loss_mode}")

    for _ in range(local_ep):
        for x, y in iter_client_batches(train_loader, device):
            logits = net(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return state_dict_to_cpu(net.state_dict())


def _local_train_noisy(
    net_student,
    net_teacher,
    train_loader,
    device,
    lr,
    local_ep,
    cls_num_list,
    w_kd,
    y_override_map=None,
    kd_idx_set=None,
    teacher_temp=0.8,
    hard_loss_mode="la",   # "la" or "ce"
    use_kd=True,
):
    net_student.train()
    net_teacher.eval()

    opt = torch.optim.Adam(net_student.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=5e-4)

    if hard_loss_mode == "la":
        criterion_hard = LogitAdjust(cls_num_list=cls_num_list, device=device)
        criterion_kd = LA_KD(cls_num_list=cls_num_list, device=device)
    elif hard_loss_mode == "ce":
        criterion_hard = nn.CrossEntropyLoss()
        criterion_kd = CE_KD()
    else:
        raise ValueError(f"Unknown hard_loss_mode: {hard_loss_mode}")

    for _ in range(local_ep):
        for x, y_noisy, idx in iter_client_batches_with_idx(train_loader, device):
            y_target = y_noisy.clone()
    
            if y_override_map is not None:
                idx_np = idx.detach().cpu().numpy().astype(np.int64)
                y_rep = [
                    int(y_override_map.get(int(i), int(y_noisy[j].item())))
                    for j, i in enumerate(idx_np)
                ]
                y_target = torch.tensor(y_rep, device=device, dtype=torch.long)

            logits = net_student(x)

            if not use_kd:
                loss = criterion_hard(logits, y_target)
            else:
                with torch.no_grad():
                    teacher_logits = net_teacher(x)
                    soft_label = torch.softmax(teacher_logits / teacher_temp, dim=1)

                if kd_idx_set is None:
                    loss = criterion_kd(logits, y_target, soft_label, float(w_kd))
                else:
                    kd_mask = torch.tensor(
                        [int(i.item()) in kd_idx_set for i in idx],
                        device=device,
                        dtype=torch.bool,
                    )

                    if kd_mask.all():
                        loss = criterion_kd(logits, y_target, soft_label, float(w_kd))
                    elif (~kd_mask).all():
                        loss = criterion_hard(logits, y_target)
                    else:
                        loss_kd = criterion_kd(
                            logits[kd_mask],
                            y_target[kd_mask],
                            soft_label[kd_mask],
                            float(w_kd),
                        )
                        loss_hard = criterion_hard(
                            logits[~kd_mask],
                            y_target[~kd_mask],
                        )
                        frac_kd = float(kd_mask.float().mean().item())
                        loss = frac_kd * loss_kd + (1.0 - frac_kd) * loss_hard

            opt.zero_grad()
            loss.backward()
            opt.step()

    return state_dict_to_cpu(net_student.state_dict())


# ============================================================
# Relabel helper
# ============================================================
@torch.no_grad()
def _refresh_relabels_for_noisy_clients(
    *,
    num_classes,
    clean_cids,
    noisy_cids,
    clean_teacher_net,
    train_loaders_plain,
    device,
    weight_mode="n",
    mode="agree",
    margin_thr=0.2,
    kd_on="all",   # "all" or "disagree"
    feats_clean=None,
    eigs_clean=None,
    y_override_map=None,
    kd_idx_set=None,
    prev_override_map=None,
    flip_rate_hist=None,
):
    if feats_clean is None:
        feats_clean = {}
    if eigs_clean is None:
        eigs_clean = {}
    if y_override_map is None:
        y_override_map = {}
    if kd_idx_set is None:
        kd_idx_set = {}
    if prev_override_map is None:
        prev_override_map = {}

    # recompute clean eigens from clean teacher
    for cid in clean_cids:
        feats_clean[cid] = extract_backbone_features(clean_teacher_net, train_loaders_plain[cid], device)
        eigs_clean[cid] = compute_feats_eigen(feats_clean[cid])


    for cid in noisy_cids:

        feats_noisy = extract_backbone_features(clean_teacher_net, train_loaders_plain[cid], device)

        res = relabel_noisy_samples_from_clean_dict(
            feats_noisy=feats_noisy,
            eigs_clean_by_client=eigs_clean,
            clean_cids=clean_cids,
            num_classes=num_classes,
            weight_mode=weight_mode,
            mode=mode,
            margin_thr=margin_thr,
            fill_with_original=True,
        )

        idx = res["idx"]
        y_svd = res["y_pred"]
        agree = res["agree"]

        cur_override_map = {int(i): int(y) for i, y in zip(idx.tolist(), y_svd.tolist())}

        flip_t, n_flip, n_total = compute_flip_rate_from_maps(
            prev_override_map.get(cid, None),
            cur_override_map,
        )
        flip_rate_hist.setdefault(cid, []).append(float(flip_t))
        print(f"[cid {cid}] flip_t={flip_t:.6f} ({n_flip}/{n_total})")

        prev_override_map[cid] = cur_override_map
        y_override_map[cid] = cur_override_map

        if kd_on == "disagree":
            kd_idx_set[cid] = set(idx[~agree].astype(np.int64).tolist())
        else:
            kd_idx_set[cid] = set(idx.astype(np.int64).tolist())

        evaluate_relabeling(
            y_clean=res["y_clean"],
            y_noisy=res["y_noisy"],
            y_after=y_svd,
            relabeled_mask=(y_svd != res["y_noisy"]),
            stage_name=f"Client {cid} - SVD",
        )

    return {
        "feats_clean": feats_clean,
        "eigs_clean": eigs_clean,
        "y_override_map": y_override_map,
        "kd_idx_set": kd_idx_set,
        "prev_override_map": prev_override_map,
        "flip_rate_hist": flip_rate_hist,
    }


# ============================================================
# Main strategy
# ============================================================
def run_fedsir(
    cfg,
    gamma_s,
    num_classes,
    base_state,
    train_loaders_train,
    train_loaders_plain,
    test_loader,
    cfg_identification,
    save_dir,
    device,
):
    """
    Hybrid:
      2) Build initial teacher/global model from clean-client aggregate
      3) Periodically relabel noisy clients with eigen-based relabeling
      4) Keep Stage 2 training:
           - clean: LogitAdjust
           - noisy: LA_KD
           - aggregate: DaAgg
    """
    os.makedirs(save_dir, exist_ok=True)

    rounds = int(getattr(cfg, "rounds", 100))
    local_ep = int(getattr(cfg, "local_ep", getattr(cfg, "local_epochs", 1)))
    lr = float(getattr(cfg, "lr", getattr(cfg, "base_lr", 3e-4)))

    begin = int(getattr(cfg, "begin", getattr(cfg, "kd_begin", 1)))
    end = int(getattr(cfg, "end", getattr(cfg, "kd_end", 49)))
    a = float(getattr(cfg, "a", getattr(cfg, "kd_a", 0.8)))
    teacher_temp = float(getattr(cfg, "teacher_temp", 0.8))

    relabeling_rounds = int(getattr(cfg, "relabeling_rounds", 1))
    kd_on = str(getattr(cfg, "kd_on", "all"))  # "all" or "disagree"

    weight_mode = str(getattr(cfg, "svd_weight_mode", "n_gap_s1"))
    mode = str(getattr(cfg, "svd_mode", "agree"))
    margin_thr = float(getattr(cfg, "svd_margin_thr", 0.2))


    aggregation_mode = str(getattr(cfg, "aggregation_mode", "daagg"))
    clean_loss_mode = str(getattr(cfg, "clean_loss_mode", "la"))
    noisy_loss_mode = str(getattr(cfg, "noisy_loss_mode", "lakd"))
    use_relabel = bool(getattr(cfg, "use_relabel", True))
    use_kd = bool(getattr(cfg, "use_kd", True))

    user_ids = list(range(cfg.num_users))
    dict_len = [len(train_loaders_train[i].dataset) for i in user_ids]

    print(
        f"[FedSIR] start | rounds={rounds} local_ep={local_ep} lr={lr:.6g} "
        f"Relabeling Rounds={relabeling_rounds} kd_on={kd_on}"
    )

    # ------------------------------------------------------------
    # 1) Identification 
    # ------------------------------------------------------------
    print("[FedSIR] identification ...")
    client_states, client_weights, clean_cids, noisy_cids = identification_process(
        cfg=cfg,
        num_classes=num_classes,
        gamma_s=gamma_s,
        train_loaders_train=train_loaders_train,
        cfg_identification=cfg_identification,
        base_state=base_state,
        save_dir=save_dir,
        device=device,
    )

    clean_cids = sorted(clean_cids)
    noisy_cids = sorted(noisy_cids)

    print(f"[FedSIR] clean_cids={clean_cids}")
    print(f"[FedSIR] noisy_cids={noisy_cids}")

    if len(clean_cids) == 0:
        raise RuntimeError("Identification produced zero clean clients. FedSIR cannot proceed.")

    # ------------------------------------------------------------
    # 2) Build clean aggregate
    # ------------------------------------------------------------
   
    client_cls_counts = {cid: collect_client_class_counts(train_loaders_train[cid], num_classes, device) for cid in user_ids}

    clean_states = [client_states[cid] for cid in clean_cids]
    clean_weights = [client_weights[cid] for cid in clean_cids]
    clean_agg_state = weighted_average_state_dicts(clean_states, clean_weights)

    # netglob is the model that will enter stage-2-like training
    netglob = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    netglob.load_state_dict(clean_agg_state, strict=True)
    netglob.eval()

    # separate clean teacher model for relabeling refresh
    global_net_clean = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    global_net_clean.load_state_dict(clean_agg_state, strict=True)
    global_net_clean.eval()

    feats_clean = {}
    eigs_clean = {}
    y_override_map = {cid: None for cid in user_ids}
    kd_idx_set = {cid: None for cid in user_ids}

    prev_override_map = {cid: None for cid in noisy_cids}
    flip_rate_hist = {cid: [] for cid in noisy_cids}

    # ------------------------------------------------------------
    # 3) FedSIR Stage 2 style loop
    # ------------------------------------------------------------
    for r in range(rounds):
        print(f"\n========== [FedSIR] Round {r+1}/{rounds} ==========")

        # refresh relabeling periodically using current clean aggregate
        if use_relabel and ((r > 0) and (r % relabeling_rounds == 0)):
            print("[FedSIR] refreshing relabels ...")
            relabel_pack = _refresh_relabels_for_noisy_clients(
                num_classes=num_classes,
                clean_cids=clean_cids,
                noisy_cids=noisy_cids,
                clean_teacher_net=global_net_clean,
                train_loaders_plain=train_loaders_plain,
                device=device,
                weight_mode=weight_mode,
                mode=mode,
                margin_thr=margin_thr,
                kd_on=kd_on,
                feats_clean=feats_clean,
                eigs_clean=eigs_clean,
                y_override_map=y_override_map,
                kd_idx_set=kd_idx_set,
                prev_override_map=prev_override_map,
                flip_rate_hist=flip_rate_hist,
            )
            feats_clean = relabel_pack["feats_clean"]
            eigs_clean = relabel_pack["eigs_clean"]
            y_override_map = relabel_pack["y_override_map"]
            kd_idx_set = relabel_pack["kd_idx_set"]
            prev_override_map = relabel_pack["prev_override_map"]
            flip_rate_hist = relabel_pack["flip_rate_hist"]

        elif not use_relabel:
            for cid in noisy_cids:
                y_override_map[cid] = None
                kd_idx_set[cid] = None if kd_on == "all" else set()

        w_locals = []
        clean_states_round = []
        clean_weights_round = []

        weight_kd = get_current_consistency_weight(r, begin, end) * a

        teacher_net = copy.deepcopy(netglob).to(device)
        # teacher_net = copy.deepcopy(global_net_clean).to(device)
        teacher_net.eval()

        for cid in user_ids:
            net_local = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
            net_local.load_state_dict(copy.deepcopy(netglob.state_dict()), strict=True)

            if cid in clean_cids:
                print(f"[Round {r+1}] client {cid} train (clean)")
                w_local = _local_train_clean(
                    net=net_local,
                    train_loader=train_loaders_train[cid],
                    device=device,
                    lr=lr,
                    local_ep=local_ep,
                    cls_num_list=client_cls_counts[cid],
                    loss_mode=clean_loss_mode,
                )
                clean_states_round.append(copy.deepcopy(w_local))
                clean_weights_round.append(len(train_loaders_train[cid].dataset))

            else:
                print(f"[Round {r+1}] client {cid} train (noisy)")
                noisy_hard_loss_mode = "la" if noisy_loss_mode in ["lakd", "la"] else "ce"
                noisy_use_kd = use_kd and (noisy_loss_mode in ["lakd", "cekd"])

                w_local = _local_train_noisy(
                    net_student=net_local,
                    net_teacher=teacher_net,
                    train_loader=train_loaders_train[cid],
                    device=device,
                    lr=lr,
                    local_ep=local_ep,
                    cls_num_list=client_cls_counts[cid],
                    w_kd=weight_kd,
                    y_override_map=y_override_map.get(cid, None) if use_relabel else None,
                    kd_idx_set=kd_idx_set.get(cid, None) if noisy_use_kd else None,
                    teacher_temp=teacher_temp,
                    hard_loss_mode=noisy_hard_loss_mode,
                    use_kd=noisy_use_kd,
                )

            w_locals.append(copy.deepcopy(w_local))
        # daagg aggregation
        global_state = aggregate_models(
            w_locals=w_locals,
            dict_len=dict_len,
            clean_clients=clean_cids,
            noisy_clients=noisy_cids,
            mode=aggregation_mode,
        )

        netglob.load_state_dict(copy.deepcopy(global_state), strict=True)
        netglob.eval()

        # update clean aggregate used for future relabeling
        if len(clean_states_round) > 0:
            clean_agg_state = weighted_average_state_dicts(clean_states_round, clean_weights_round)
            global_net_clean.load_state_dict(copy.deepcopy(clean_agg_state), strict=True)
            global_net_clean.eval()

        # --------------------------------------------------------
        # evaluation
        # --------------------------------------------------------
        yt, yp, per_loss = predict(netglob, test_loader, device)
        m = per_class_metrics(yt, yp, num_classes=num_classes)
        m["loss"] = float(per_loss.mean())

        ci_deltas = bootstrap_ci(
            yt, yp, per_loss,
            num_classes=num_classes,
            B=1000,
            seed=123,
            alpha=0.05,
        )

        save_metrics_report_txt_with_ci(
            os.path.join(save_dir, "results.txt"),
            "TEST performance: net_final",
            m,
            num_classes,
            ci_deltas=ci_deltas,
            extra_lines=[
                f"================ Round {r+1} ================",
                f"Clients: {gamma_s}",
                "Stage 2",
                f"selected noisy clients: {noisy_cids} | selected clean clients: {clean_cids}",
                f"weight_kd: {weight_kd:.6f}  (a={a}, ramp=[{begin},{end}])",
                f"kd_on: {kd_on}",
                f"relabeling_rounds: {relabeling_rounds}",
            ],
        )

    print("[FedSIR] done")