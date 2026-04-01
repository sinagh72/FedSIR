# fedprox.py
import os
from model.build_model import build_backbone_model
from model.trainer import predict, train_model
from algorithm.helper import state_dict_to_cpu, weighted_average_state_dicts
from utils.metrics import bootstrap_ci, per_class_metrics, save_metrics_report_txt_with_ci


def run_fedprox(cfg, gamma_s, num_classes, base_state, train_loaders_train, test_loader, cfg_train, save_dir, device, mu=0.2):
    print(f"\n==================  FedProx Training ({cfg.rounds} rounds, mu={mu}) ==================")

    # ----- initialize global model -----
    global_net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
    global_net.load_state_dict(base_state, strict=True)
    global_state = state_dict_to_cpu(global_net.state_dict())

    for r in range(cfg.rounds):
        print(f"\n========== Round {r+1}/{cfg.rounds} ==========")

        # Snapshot of global params for FedProx penalty (IMPORTANT)
        # Use named_parameters keys to align with trainer _prox_term
        global_params = {name: p.detach().cpu().clone() for name, p in global_net.named_parameters()}

        local_states = []
        local_weights = []

        for cid in range(cfg.num_users):
            net = build_backbone_model(cfg.model_name, cfg.pretrained, num_classes).to(device)
            net.load_state_dict(global_state, strict=True)

            print(f"\n--- Client {cid} Training (Round {r+1}) ---")

            net, _ = train_model(
                net,
                train_loaders_train[cid],
                device,
                cfg_train,
                cfg,
                label_mode="noisy",
                mu=mu,
                global_params=global_params,
            )

            local_states.append(state_dict_to_cpu(net.state_dict()))
            local_weights.append(len(train_loaders_train[cid].dataset))

        # ----- aggregation (same as FedAvg) -----
        global_state = weighted_average_state_dicts(local_states, local_weights)

        # ----- evaluation (same as FedAvg) -----
        global_net.load_state_dict(global_state, strict=True)
        global_net.eval()

        yt, yp, per_loss = predict(global_net, test_loader, device)

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
                f"FedProx mu: {mu}",
            ],
        )