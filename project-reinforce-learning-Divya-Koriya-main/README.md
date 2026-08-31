# 📦 Predicting E-Commerce Delivery Delays — Explainable Machine Learning and Ensemble Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-yellow)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

</div>

---

## 🎓 Academic Information

| Field | Details |
|:---|:---|
| **University** | SRH University Leipzig |
| **Student Name** | Kajal Anandani |
| **Matriculation Number** | 100001782 |
| **Subject** | Master Thesis |
| **Project** | Predicting E-Commerce Delivery Delays Using Explainable Machine Learning and Ensemble Learning |
| **Date** | August 2026 |

---

## 📌 Project Overview

This project builds a **cost-sensitive, leakage-aware machine learning pipeline** to predict late e-commerce deliveries using the DataCo Supply Chain Dataset (180,519 orders).

The pipeline:
- 🔍 Detects and removes **data leakage** (features only known after delivery)
- ⚖️ Balances the training data with **SMOTE**
- 🧠 Trains and compares **Random Forest, LightGBM, and CatBoost**
- 🤝 Combines them via **Voting** and **Stacking** ensembles
- 💰 Evaluates models by **actual business cost**, not just F1 score
- 🔎 Explains predictions using **SHAP**
- 📊 Validates robustness via **bootstrap resampling** (1,000 iterations)

> **Key Principle:** This project treats leakage-free evaluation and cost-sensitive decision-making as first-class concerns — not an afterthought to accuracy metrics.

---

## 🗂️ Repository Structure

```
Code/
│
├── kajal_thesis.py    # Complete pipeline (exported from Google Colab)
└── README.md            # This file
```

---

## 🔗 Original Notebook

The original Google Colab notebook (interactive version with all outputs) is available at:
https://colab.research.google.com/drive/1iJfIUQXHBm-Mdur4jDuVzzoFNRtCDsb9

---

## 🌍 Data Source

| Field | Details |
|:---|:---|
| **Dataset** | DataCo Supply Chain Dataset (DataCoSupplyChainDataset.csv) |
| **Source** | [Kaggle — DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) |
| **Records** | 180,519 individual orders |
| **Initial features** | 53 |
| **Access method in code** | `google.colab.files.upload()` — interactive upload inside Google Colab |

---

## 🔧 Leakage Features Removed

| Feature | Why it leaks |
|:---|:---|
| Delivery Status | Directly encodes the target (on-time vs. late) |
| Days for Shipping (Real) | Only known after delivery completes |
| Shipping Date | Assigned during fulfillment, not at order time |
| Shipping Duration Differential | Computed from post-delivery data |

---

## 🤖 Models and Configuration

| Model | Key Hyperparameters |
|:---|:---|
| Random Forest | `n_estimators=100`, `random_state=42` |
| LightGBM | `n_estimators=100`, `random_state=42` |
| CatBoost | `iterations=200`, `random_state=42` |
| Voting Ensemble | Equal weight (1/3 each) across the three base learners |
| Stacking Ensemble | Logistic Regression meta-learner (`scikit-learn StackingClassifier`), default parameters |
| PCA configuration | `n_components=0.90` (90% variance retained), `random_state=42` |
| SMOTE | `random_state=42`, applied to training data only |

Feature sets evaluated: **All Features (49)**, **Top 25**, **Top 15**, **PCA (90% variance)** — ranked via Random Forest feature importance.

---

## 📈 Training / Evaluation Results

| Metric | Random Forest (Top 15) | Voting Ensemble | Stacking Ensemble |
|:---|:---:|:---:|:---:|
| F1 Score | 0.8865 | 0.7709 | **0.9126** |
| ROC-AUC | 0.9622 | 0.9259 | **0.9699** |
| Operational Cost (36,104 test orders) | \$7,859.99 | \$13,790.47 | **\$5,616.82** |

| Finding | Value |
|:---|:---|
| Leakage-induced F1 inflation | 11.64 percentage points (1.0000 → 0.8836) |
| Lowest-cost threshold (evaluated grid: 0.25–0.75) | **0.25** |
| Cost reduction vs. default (0.50) threshold | 30.2% |
| Cost reduction, Stacking vs. Voting | 59.3% |
| Bootstrap 95% CI (Stacking ROC-AUC, 1,000 resamples) | [0.9684, 0.9714] |
| Top SHAP feature | Scheduled shipping duration (mean SHAP value: 0.1257) |

These values match Chapter 4 (Results) of the thesis PDF exactly.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.x (developed and run in Google Colab's default runtime)
- pip

### Install Dependencies

The notebook installs its extra dependencies directly in the first cell:
```bash
pip install -q imbalanced-learn lightgbm catboost shap
```
`pandas`, `numpy`, `matplotlib`, and `scikit-learn` come pre-installed in Google Colab.

To run outside Colab, install everything explicitly:
```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn lightgbm catboost shap
```

---

## ▶️ How to Run

1. Open `kajal_thesis.py` in Google Colab (or import it as a notebook).
2. Run the first cell to install dependencies.
3. When prompted by `files.upload()`, upload the DataCo Supply Chain Dataset CSV.
4. Run all cells top to bottom.
5. The script reproduces, in order: leakage removal → preprocessing/SMOTE → feature selection → model training (RF/LightGBM/CatBoost) → ensemble construction (Voting/Stacking) → cost-sensitive evaluation → threshold optimization → SHAP analysis → bootstrap validation.

---

## 🔁 Reproducibility

Every random process (train-test split, SMOTE, model initialization, PCA, bootstrap resampling) uses a fixed seed: `random_state=42`.

---

## 🗺️ Code Section → Thesis Location Map

| Code Section | Produces | Thesis Location |
|:---|:---|:---|
| Dataset upload | Raw data load, 180,519 records | Ch. 3.2 Dataset Selection |
| Leaky vs. clean dataset prep | 4 leakage features removed | Ch. 3.4 Leakage Detection |
| Leakage impact visualization | F1 1.0000 → 0.8836 plot | Figure 4.1, Ch. 4.1 |
| Random Forest training | RF F1/ROC-AUC across feature sets | Table 4.1 |
| Ensemble model building | Voting & Stacking training | Ch. 3.9, Table 4.2 |
| Confusion matrix creation | Stacking confusion matrix | Figure 4.2 |
| Business cost estimation | Cost comparison (\$5,616.82 etc.) | Figure 4.4, Ch. 4.7 |
| Threshold evaluation | 0.25 lowest-cost threshold, 30.2% reduction | Figure 4.5, Ch. 4.8 |
| Bootstrap confidence intervals | 95% CI [0.9684, 0.9714] | Figure 4.6, Ch. 4.9 |
| SHAP analysis | Feature importance (0.1257, 0.0796) | Figure 4.3, Ch. 4.6 |

---

## 📚 Key References

- Kaufman, S. et al. (2012) — data leakage taxonomy
- Breiman, L. (2001) — Random Forest
- Ke, G. et al. (2017) — LightGBM
- Prokhorenkova, L. et al. (2018) — CatBoost
- Lundberg, S. & Lee, S. (2017) — SHAP
- Chawla, N. et al. (2002) — SMOTE

(Full citation list in the thesis Bibliography.)

---

## 🧾 Generative AI Disclosure

Generative AI tools were used during development for conceptual explanation, literature search/analysis, methodology guidance, code generation, visualization generation, and text structuring/editing — as disclosed in the thesis Affidavit and List of Generative AI Tool Usages.

---

## 📧 Contact

For questions about this code, contact Kajal Anandani via SRH University Leipzig.
