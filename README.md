# 📝 Predicting Customer Review Ratings

**Tilburg University · MSc Data Science and Society · Master's Thesis 2024**

A thesis project that predicts star ratings from customer review text using transformer models (BERT, roBERTa) and a Random Forest baseline.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Transformers-orange.svg)
![Thesis](https://img.shields.io/badge/Thesis-MSc%20DSS-green.svg)

## 📋 Overview

This thesis tackles the **review rating prediction** task: given the text of a customer review, predict its star rating (e.g., 1–5 stars). The project compares:

| Model | Role |
|-------|------|
| **Random Forest** | Baseline (traditional ML) |
| **BERT** | Transformer-based approach |
| **roBERTa** | Optimized RoBERTa variant |

## 🏗️ Project Structure

```
├── Random Forest - Baseline/      # Baseline model implementation
├── BERT Hyperparameter Search/     # BERT hyperparameter tuning
├── BERT runs/                      # BERT model training runs
├── roBERTa Hyperparameter Search/  # roBERTa hyperparameter tuning
├── roBERTa runs/                   # roBERTa model training runs
├── Images/                         # Plots and visualizations
├── Tilburg_University_DSS_Masters_Thesis_Ivaylo_Papazov_2024.pdf
└── README.md
```

## 📖 Thesis

For the full methodology, experiments, and conclusions, see:

**[Tilburg_University_DSS_Masters_Thesis_Ivaylo_Papazov_2024.pdf](Tilburg_University_DSS_Masters_Thesis_Ivaylo_Papazov_2024.pdf)**

## 🚀 Navigating the Repository

| Goal | Location |
|------|----------|
| Hyperparameter tuning procedures | `BERT Hyperparameter Search/`, `roBERTa Hyperparameter Search/` |
| Model training runs | `BERT runs/`, `roBERTa runs/` |
| Generated plots and figures | `Images/` |
| Full thesis document | PDF file in root |

## 👤 Author

**Ivaylo Papazov** – Tilburg University, MSc Data Science and Society

## 📄 License

Academic work. Please cite if used.

---

⭐ If you find this project useful, please consider giving it a star!
