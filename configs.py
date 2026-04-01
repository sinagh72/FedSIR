from dataclasses import dataclass

@dataclass
class ExpConfig:
    # dataset
    dataset: str = "cifar10"

    iid: bool = False
    non_iid_prob_class: float = 1.0
    alpha_dirichlet: float = 2.0
    num_users: int = 10

    # noise
    level_n_system: float = 0.9
    noise_p: float = 0.9
    noise_type: str = "sym"
    save_dir: str = "./results"

    # model
    model_name: str = "resnet18"
    pretrained: bool = True
    batch_size: int = 64
    
    # FL
    fl_strategy: str = "fedsir"   # "fedavg", fedprox, "pruning", fedsir
    rounds: int = 100

    # relabel
    identification_epochs: int = 5
    identification_lr: float = 5e-5
    identification_weight_decay: float = 5e-2
    
    # kd
    kd_begin = 2
    kd_end = 49
    kd_a = 0.2
    kd_temp = 1.0
    # training param
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 5e-4
    optimizer: str = "Adam"
    seed: int = 42

    relabeling_rounds: int = 20            # how often to relabels during training
    aggregation_mode: str = "daagg"        # "daagg" or "weighted_avg"
    clean_loss_mode: str = "la"            # "la" or "ce"
    noisy_loss_mode: str = "lakd"          # "lakd", "la", "cekd", "ce"
    use_relabel: bool = True
    use_kd: bool = True
    kd_on: str = "disagree"                # "all" or "disagree"
    svd_mode: str = "agree"
