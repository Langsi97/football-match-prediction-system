# ⚽ Football Match Prediction System  
### Production-Grade Machine Learning Pipeline for Sports Analytics

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Production%20Ready-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Overview

This project is a **production-grade end-to-end machine learning system** designed to predict football match outcomes in the Belgian Jupiler Pro League.

Unlike typical notebook-based projects, this system is built with **real-world ML engineering practices**, including:

- Modular architecture  
- Config-driven pipelines  
- Reproducible workflows  
- Clean separation of data, features, and models  

Beyond prediction, the system evaluates:

- 📊 **Model-implied probabilities**
- 💰 **Bookmaker odds**
- ⚠️ **Market inefficiencies and margins (vig)**

---

## 🎯 Business & Research Impact

This project addresses a key question:

> Can machine learning outperform bookmakers?

### Findings:
- Prediction accuracy is inherently limited (~45–54%)
- Bookmakers consistently maintain profit margins (vig)
- Betting markets are **efficient but biased**

👉 This project demonstrates why **“the house always wins”**

---
Raw Data → Validation → Standardization → Merge → Preprocessing
→ Feature Engineering → Modeling → Evaluation → Prediction → Odds Analysis

## 🏗️ System Architecture

### 📌 Design Principles
- No data leakage
- Pre-match features only
- Time-aware splitting
- Reproducibility first

---

## 📂 Project Structure
```
football-match-prediction-system/
│
├── config/              # Pipeline configuration (YAML)
│
├── data/
│   ├── raw/             # Raw datasets (local only)
│   ├── interim/         # Merged datasets
│   ├── processed/       # Final modeling datasets
│   └── external/        # League position data
│
├── scripts/             # Pipeline runners
│
├── src/
│   ├── data/            # Data engineering modules
│   ├── features/        # Feature engineering
│   ├── models/          # Model training (planned)
│   └── utils/           # Config & path utilities
│
├── notebooks/           # Research & experimentation
│
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Pipeline (Production Flow)

### 1️⃣ Data Loading
```
python -m scripts.run_data_load
```
### 2️⃣ Schema Validation
```
python -m scripts.run_schema_check
```

### 3️⃣ Season Merging
```
python -m scripts.run_merge_data
```

### 4️⃣ Preprocessing
```
python -m scripts.run_preprocess_matches
```

### 5️⃣ Dataset Enrichment
```
python -m scripts.run_enrich_with_positions
```

## 📊 Dataset
Seasons Covered
2019/2020 → 2023/2024
Final Dataset
✅ 1390 matches
✅ 51 features
✅ Pre-match only (no leakage)
Key Features
Team ratings
Match statistics
League positions
Temporal features

## 🧠 Feature Engineering (Core Strength)
🔥 Rolling Features (Upcoming)
3-match rolling averages:
Goals
Shots
Cards
Corners
📈 Team Form
Last ≤5 matches
Normalized performance
🏠 Home Advantage
Last ≤5 home games
Adaptive scoring

## 🤖 Models
Logistic Regression
Random Forest
XGBoost
SVM
Naive Bayes
Evaluation Metrics
Accuracy
F1 Score (weighted)
AUC
Log Loss
Brier Score
💰 Betting Intelligence Layer

This is what makes the project unique.

## 📌 Capabilities
Convert probabilities → odds
Compare vs bookmaker odds
Compute vig (margin)
Detect bias (favorite–longshot)

## 📉 Profit Simulation
Strategy testing
Real-world betting scenarios

## 🛠️ Tech Stack

Python
Data	Pandas, NumPy
	Scikit-learn, XGBoost
	Matplotlib, Seaborn
Pipeline	Custom modular system
	Streamlit (planned), Azure (planned)
  
## 🔒 Engineering Best Practices

✔ Config-driven pipelines
✔ No raw data in Git
✔ No model artifacts tracked
✔ Modular codebase
✔ Reproducible workflows
✔ Clean Git history (no large files)

## 🚧 Roadmap
### 🔜 Phase 2
Rolling feature pipeline
Form & home advantage modules
### 🔜 Phase 3
Optuna hyperparameter tuning
Model selection automation
### 🔜 Phase 4
Streamlit dashboard
Prediction UI
### 🔜 Phase 5
MLflow tracking
Dockerization
Azure deployment

## 🧪 Example Output
```Model Prediction
Home Win: 52%
Draw: 24%
Away Win: 24%
```
```Converted Odds
Home: 1.92
Draw: 4.16
Away: 4.16
```
## Insight
Bookmaker Odds > Model Odds → Overpriced → Not profitable

## 📌 Key Insight

Even with advanced machine learning, football remains highly unpredictable —
and bookmaker margins ensure long-term profitability.

## 👤 Author

Langsi Ambe Revelation
MSc Statistics & Data Science — Belgium

AI & Machine Learning Engineer
Sports Analytics Enthusiast
Focused on production-ready ML systems

## ⭐ Why This Project Stands Out
Not a notebook → real ML system
Business + technical + research combined
Production engineering mindset
End-to-end pipeline design
🔗 Let’s Connect

If you're a recruiter or hiring manager:

## 👉 This project demonstrates:

Production ML skills
Strong data engineering
Real-world problem solving

Feel free to reach out 🚀
