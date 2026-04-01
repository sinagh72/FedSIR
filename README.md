

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
  [
  z_i = f_\phi(x_i)
  ]
* Compute class-wise SVD
* Build similarity matrix:
  [
  S_{c,c'} = |v_c^T v_{c'}|
  ]
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

  * FedAvg
  * FedProx
  * FedNoRo
  * FedELC
  * FedCorr

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
python main.py \
    --dataset cifar10 \
    --num_clients 10 \
    --noise_rate 0.5 \
    --dirichlet_alpha 0.5 \
    --model resnet18
```

---

## ⚙️ Key Hyperparameters

| Parameter    | Description                            |
| ------------ | -------------------------------------- |
| `alpha`      | Dirichlet non-IID strength             |
| `noise_rate` | Label corruption level                 |
| `E1`         | Stage I epochs (client identification) |
| `E2`         | Local training epochs                  |
| `R`          | Relabeling interval                    |
| `T`          | Total communication rounds             |
| `w_KD`       | KD weight                              |
| `tau`        | Logit adjustment strength              |

---

## 📁 Project Structure

```
fedsir/
│── main.py
│── models/
│── data/
│── utils/
│── configs/
│── training/
│── spectral/
│── README.md
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
* Performance depends on **feature quality in early training**

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

