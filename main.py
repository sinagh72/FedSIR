from configs import ExpConfig
from dataset.dataset import get_dataset
from utils.set_seed import set_all_seeds
from run import run_fl

if __name__ == "__main__":
    cfg = ExpConfig()
    set_all_seeds(cfg.seed)
    for alpha in [2.0]:
        cfg.alpha_dirichlet = alpha
        dataset_train, dataset_test, dict_users, num_classes = get_dataset(
            dataset_name=cfg.dataset,
            num_users=cfg.num_users,
            iid=cfg.iid,
            non_iid_prob_class=cfg.non_iid_prob_class,
            alpha_dirichlet=alpha,
            seed=cfg.seed,
        )

        for i in range(7):
            noise_p = round(0.3 + 0.1 * i, 1)
            cfg.noise_p = noise_p

            run_fl(
                cfg=cfg,
                dataset_train=dataset_train,
                dataset_test=dataset_test,
                dict_users=dict_users,
                num_classes=num_classes,
            )