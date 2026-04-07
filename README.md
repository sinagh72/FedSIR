

## 🧠 FedSIR: Spectral Client Identification and Relabeling for Federated Learning with Noisy Labels

📄 Paper: 
*(CVPR 2026 submission)*

---

## 🚀 Overview

Federated Learning (FL) enables collaborative model training without sharing raw data. However, **noisy labels across clients severely degrade performance**, especially under non-IID data distributions.

We propose **FedSIR**, a novel multi-stage framework that leverages the **spectral structure of feature representations** to:

* Identify **clean vs. noisy clients**
* Perform **reliable label correction**
* Enable **robust federated optimization**

Unlike prior work, FedSIR does **not rely on loss dynamics or clean validation data**, but instead uses **intrinsic feature geometry**.

---

## 🔑 Key Contributions

* 🔍 **Spectral Client Identification**
  Detect clean/noisy clients using class-wise feature subspaces and similarity statistics.

* 🔄 **Spectral Relabeling**
  Correct noisy labels using:

  * Dominant class directions
  * Residual subspaces
    with a **conservative agreement rule**

* ⚖️ **Noise-Aware Optimization**

  * Logit-Adjusted Loss (LA)
  * Knowledge Distillation (KD)
  * Distance-aware aggregation (DaAgg)

* 📊 **Strong Empirical Results**
  Outperforms SOTA methods under:

  * High label noise (30%–90%)
  * Strong non-IID settings

---

## 🧩 Method Overview

FedSIR consists of **three stages**:

### 1️⃣ Client Identification

* Extract feature representations:
  $$
  z_i = f_\phi(x_i)
  $$
* Compute class-wise SVD
* Build similarity matrix:
  [S_{c,c'} = |v_c^T v_{c'}|]
* Use statistics (mean & energy) + GMM to classify:

  * ✅ Clean clients
  * ❌ Noisy clients

---

### 2️⃣ Spectral Relabeling

From clean clients, construct:

* Dominant direction: ( \bar{v}_c^{(r)} )
* Residual subspace: ( \bar{V}_c^{(n)} )

For each sample:

* Alignment score:
  [
  S^{(r)}(i,c)
  ]
* Residual score:
  [
  S^{(n)}(i,c)
  ]

✔ Relabel only if:
[
\hat{y}^{(r)} = \hat{y}^{(n)}
]

👉 This ensures **high-confidence corrections only**

---

### 3️⃣ Noise-Aware Training

#### Clean Clients:

* Logit-adjusted CE:
  [
  \mathcal{L}_{LA} = CE(f(x) + m_k, y)
  ]

#### Noisy Clients:

* Hybrid loss:
  [
  \mathcal{L}*{LA-KD} = w*{KD} \cdot KL(p | q) + (1-w_{KD}) \cdot CE
  ]

#### Aggregation:

* Distance-aware weighting:
  [
  \alpha_k \propto a_k \cdot e^{-d_k}
  ]

---

## 📊 Results

* Robust under **extreme noise (up to 90%)**
* Strong performance under **Dirichlet non-IID splits**
* Consistently outperforms:

  * [FedAvg](https://arxiv.org/abs/1602.05629)
  * [FedProx](https://arxiv.org/abs/1812.06127)
  * [RoFL](https://ieeexplore.ieee.org/document/9767609)
  * [RHFL](https://openaccess.thecvf.com/content/CVPR2022/html/Fang_Robust_Federated_Learning_With_Noisy_and_Heterogeneous_Clients_CVPR_2022_paper.html)
  * [FedLSR](https://dl.acm.org/doi/10.1145/3511808.3557654)
  * [FedCorr](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_FedCorr_Multi-Stage_Federated_Learning_for_Label_Noise_Correction_CVPR_2022_paper.html)
  * [FedNed](https://ojs.aaai.org/index.php/AAAI/article/view/29667)
  * [FedELC](https://dl.acm.org/doi/10.1145/3627673.3679990)
  * [FedNoRo](https://arxiv.org/abs/2305.05230)

👉 Key insight:
**Spectral structure is a reliable signal for noisy label detection in FL** 

---

## 🛠️ Installation

```bash
git clone https://github.com/repo/fedsir.git
cd fedsir

pip install uv
pip uv sync

source .venv/bin/activate
```

---

## ▶️ Training

Example (CIFAR-10):

```bash
python main.py
```

---

## ⚙️ Key Hyperparameters

### 📦 Dataset & Distribution

| Parameter            | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `dataset`            | Dataset used (e.g., CIFAR-10, CIFAR-100)                                                    |
| `iid`                | Whether data is IID across clients                                                          |
| `alpha_dirichlet`    | Dirichlet concentration parameter (controls non-IID severity; smaller = more heterogeneous) |
| `non_iid_prob_class` | Probability of assigning dominant classes to clients                                        |
| `num_users`          | Number of federated clients                                                                 |

---

### 🔊 Noise Configuration

| Parameter        | Description                       |
| ---------------- | --------------------------------- |
| `noise_p`        | Label noise ratio per client      |
| `level_n_system` | Overall system-level noise ratio  |
| `noise_type`     | Type of noise (`sym` = symmetric) |

---

### 🧠 Model

| Parameter    | Description                         |
| ------------ | ----------------------------------- |
| `model_name` | Backbone model (e.g., ResNet18)     |
| `pretrained` | Whether to use ImageNet pretraining |
| `batch_size` | Local training batch size           |

---

### 🌐 Federated Learning

| Parameter     | Description                                     |
| ------------- | ----------------------------------------------- |
| `fl_strategy` | FL method (`fedavg`, `fedprox`, `fedsir`, etc.) |
| `rounds`      | Total communication rounds (T)                  |

---

### 🔍 Stage I: Client Identification

| Parameter                     | Description                                           |
| ----------------------------- | ----------------------------------------------------- |
| `identification_epochs`       | Number of local epochs for client identification (E1) |
| `identification_lr`           | Learning rate during identification                   |
| `identification_weight_decay` | Weight decay during identification                    |

---

### 🔁 Training (Stage II & III)

| Parameter      | Description                          |
| -------------- | ------------------------------------ |
| `epochs`       | Local training epochs per round (E2) |
| `lr`           | Learning rate                        |
| `weight_decay` | Weight decay                         |
| `optimizer`    | Optimizer (e.g., Adam)               |
| `seed`         | Random seed                          |

---

### 🔄 Relabeling (Stage II)

| Parameter           | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `relabeling_rounds` | Relabeling interval (R)                                     |
| `use_relabel`       | Enable spectral relabeling                                  |
| `svd_mode`          | Relabeling strategy (`agree` = conservative agreement rule) |

---

### 📚 Knowledge Distillation (KD)

| Parameter  | Description                                    |
| ---------- | ---------------------------------------------- |
| `use_kd`   | Enable knowledge distillation                  |
| `kd_begin` | Round to start KD                              |
| `kd_end`   | Round to stop KD                               |
| `kd_a`     | KD weight ( ( w_{KD} ) )                       |
| `kd_temp`  | Temperature for soft labels                    |
| `kd_on`    | Apply KD to (`all` samples or only `disagree`) |

---

### ⚖️ Loss & Aggregation

| Parameter          | Description                                    |
| ------------------ | ---------------------------------------------- |
| `clean_loss_mode`  | Loss for clean clients (`la`, `ce`)            |
| `noisy_loss_mode`  | Loss for noisy clients (`lakd`, `cekd`, etc.)  |
| `aggregation_mode` | Aggregation method (`daagg` or `weighted_avg`) |

---

## 🔥 Notes (Important for Readers)

* `alpha_dirichlet` controls **client heterogeneity**
* `noise_p` + `level_n_system` define **label corruption severity**
* `kd_a` is your **( w_{KD} )** in the paper formulation 
* `relabeling_rounds` corresponds to **R (periodic relabeling)**
* `rounds` corresponds to **T (total FL rounds)**
---

## 📁 Project Structure

```
fedsir/
│── algorithm/
│── dataset/
│── model/
│── utils/
│── configs.py
│── main.py
│── run.py
```

---

## 🔬 Experimental Setup

* Dataset: CIFAR-10 / CIFAR-100
* Backbone: ResNet-18 (ImageNet pretrained)
* Optimizer: Adam
* Communication rounds: 100
* Noise: 30%–90% symmetric
* Clients: 10

---

## ⚠️ Limitations

* Assumes **symmetric label noise**
* Requires **sufficient class coverage** among clean clients
* Performance depends on **feature quality**

---

## 🔮 Future Work

* Extend to **asymmetric noise**
* Apply to **real-world FL datasets**
* Explore **feature-level robustness beyond vision**

---

## 📚 Citation

```bibtex
@article{fedsir2026,
  title={FedSIR: Spectral Client Identification and Relabeling for Federated Learning with Noisy Labels},
  author={Anonymous},
  journal={CVPR},
  year={2026}
}
```

---

## 🤝 Acknowledgments

This work explores a new direction in FL by leveraging **spectral geometry as supervision**, offering a principled alternative to loss-based noise handling.

---

## ⭐ If you find this useful

Please consider ⭐ starring the repo!

