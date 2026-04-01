import torch
import copy
import numpy as np


def state_dict_to_cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def weighted_average_state_dicts(state_dicts, weights):
    """
    Args:
        state_dicts: List[Dict[str, Tensor]]
        weights:     List[int or float] (same length)

    Returns:
        averaged state_dict
    """
    assert len(state_dicts) == len(weights)
    total_weight = float(sum(weights))
    assert total_weight > 0

    avg = {}
    for k in state_dicts[0].keys():
        if not torch.is_floating_point(state_dicts[0][k]):
            avg[k] = state_dicts[0][k]
            continue

        avg[k] = sum(
            (w / total_weight) * sd[k]
            for sd, w in zip(state_dicts, weights)
        )

    return avg


def _model_dist(w1, w2):
    dist_total = 0.0
    for k in w1.keys():
        t1, t2 = w1[k], w2[k]
        if not torch.is_tensor(t1) or not torch.is_tensor(t2):
            continue
        if not torch.is_floating_point(t1):
            continue
        dist_total += torch.norm(t1.float() - t2.float()).item()
    return float(dist_total)


def daagg(w_locals, dict_len, clean_clients, noisy_clients):
    """
    Assumes w_locals are aligned to client id order [0..K-1].
    """
    client_weight = np.array(dict_len, dtype=np.float64)
    client_weight = client_weight / (client_weight.sum() + 1e-12)

    distance = np.zeros(len(dict_len), dtype=np.float64)
    noisy_set = set(noisy_clients)

    for n_idx in range(len(dict_len)):
        if n_idx not in noisy_set:
            continue
        dis = []
        for c_idx in clean_clients:
            dis.append(_model_dist(w_locals[n_idx], w_locals[c_idx]))
        distance[n_idx] = float(min(dis)) if len(dis) > 0 else 0.0

    if distance.max() > 0:
        distance = distance / distance.max()

    client_weight = client_weight * np.exp(-distance)
    client_weight = client_weight / (client_weight.sum() + 1e-12)

    w_avg = copy.deepcopy(w_locals[0])
    for k in w_avg.keys():
        if torch.is_tensor(w_avg[k]) and not torch.is_floating_point(w_avg[k]):
            w_avg[k] = w_locals[0][k]
            continue
        if torch.is_tensor(w_avg[k]):
            w_avg[k] = w_locals[0][k].float() * client_weight[0]
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k].float() * client_weight[i]
    return w_avg


@torch.no_grad()
def collect_client_class_counts(train_loader, num_classes: int, device):
    """
    Build cls_num_list from client's NOISY labels.
    Expects batches as (x, y_noisy, y_clean, idx)
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in train_loader:
        y_noisy = batch[1].to(device, non_blocking=True).long()
        y_np = y_noisy.detach().cpu().numpy()
        for c in range(num_classes):
            counts[c] += int((y_np == c).sum())
    counts = np.maximum(counts, 1)
    return counts.tolist()


def iter_client_batches(train_loader, device):
    for batch in train_loader:
        x = batch[0].to(device, non_blocking=True)
        y_noisy = batch[1].to(device, non_blocking=True).long()
        yield x, y_noisy


def iter_client_batches_with_idx(train_loader, device):
    """
    Expects batches as (x, y_noisy, y_clean, idx)
    """
    for batch in train_loader:
        x = batch[0].to(device, non_blocking=True)
        y_noisy = batch[1].to(device, non_blocking=True).long()
        idx = batch[3].to(device, non_blocking=True).long()
        yield x, y_noisy, idx