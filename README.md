
# 🔥 Ensemble Project: Predicting Calorie Expenditure

This project tackles a regression problem from the **Kaggle Playground Series (S5E5)** competition. The goal is to accurately predict how many calories a person burns during physical activity based on biometric and workout-related features.

---

## 🎯 Objective

To develop and optimize models that predict calorie expenditure using a combination of:

- Biometric features (age, height, weight, gender)
- Activity features (duration, heart rate, body temperature)

---

## 📊 Methodology

### 1. Data Cleaning & Preparation

- Dropped unused or redundant columns
- Encoded categorical variables (e.g., gender)
- Handled outliers and verified unit consistency
- Engineered key features:
  - BMI = Weight / Height²
  - Interaction terms (e.g., Duration × Heart Rate)
  - Polynomial features (Duration²)

---

### 2. Model Selection & Training

- Chose five powerful regression models:
  - `RandomForestRegressor`
  - `HistGradientBoostingRegressor`
  - `XGBRegressor`
  - `LGBMRegressor`
  - `CatBoostRegressor`

- Each model was:
  - Tuned using **Optuna** (100 trials/model)
  - Evaluated using **RMSLE** with 5-fold CV
  - Tracked with **MLflow** for reproducibility

---

### 3. Ensemble Strategy

- **Optuna-weighted averaging** of 5 base models:
  - Weights were optimized to minimize RMSLE
  - Final prediction formula:
    ```
    final_pred = w1 * RF + w2 * HGB + w3 * XGB + w4 * LGBM + w5 * CatBoost
    ```

- Submission using this ensemble achieved:
  ✅ **Leaderboard RMSLE: 0.05798** (Top 9%)

---

### 4. Stacking & OOF (Advanced)

- Implemented **Out-of-Fold (OOF) stacking**:
  - Generated OOF predictions for each base model
  - Trained a **Ridge meta-model** on these as inputs
  - Produced a stacked test prediction

- Also explored:
  - Manual weight ensembles
  - Ridge + Ensemble blending
  - Weighted correlation-based stacking

---

## 📈 Results

| Model        | CV RMSLE (Validation) | Leaderboard RMSLE |
|--------------|-----------------------|--------------------|
| RandomForest | 0.063xx               | —                  |
| XGBoost      | 0.058xx               | —                  |
| CatBoost     | 0.057xx               | —                  |
| Ensemble     | 0.05750               | ✅ **0.05798**      |

---

## 📂 Project Structure

```
.
├── data
│   ├── processed
│   └── raw
├── logs
│   ├── best_params
│   ├── best_params_orig_features
│   ├── catboost_logs
│   └── mlruns
├── models
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model_fe.ipynb
│   ├── 05_feature_selection.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_ensemble_model.ipynb
│   ├── init.py
│   └── __pycache__
├── outputs
│   ├── metrics.csv
│   ├── submission_avg_ensemble.csv
│   ├── submission_baseline_catboost.csv
│   ├── submission_baseline_catboost_fe.csv
│   ├── submission_baseline_catboost_fe_updated.csv
│   ├── submission_ensemble_lgbm_xgb.csv
│   ├── submission_ensemble_optuna_v11.csv
│   ├── submission_ensemble_tuned_v10.csv
│   ├── submission_ensemble_tuned_v3.csv
│   ├── submission_ensemble_tuned_v4.csv
│   ├── submission_ensemble_tuned_v5.csv
│   ├── submission_ensemble_tuned_v6.csv
│   ├── submission_ensemble_tuned_v7.csv
│   ├── submission_ensemble_tuned_v8.csv
│   ├── submission_ensemble_tuned_v9.csv
│   ├── submission_ensemble_v1.csv
│   ├── submission_ensemble_v2.csv
│   ├── submission_lgbm_cv.csv
│   ├── submission_lgbm_v1.csv
│   ├── submission_oof_stack_ridge_v1.csv
│   ├── submission_stacked_ensemble.csv
│   ├── submission_tuned_weighted_ensemble.csv
│   ├── submission_v1.csv
│   └── submission_weighted_ensemble.csv
├── README.md
├── requirements.txt
└── src
    ├── config.py
    ├── __init__.py
    ├── __pycache__
    └── utils.py

```
### 📌 Conclusion

This project was a deep dive into building a full machine learning pipeline to predict calorie expenditure. It combined rigorous feature engineering, five base models fine-tuned with Optuna, and multiple ensemble strategies including Ridge stacking and out-of-fold (OOF) meta-modeling.

Key takeaways from this project:

- ✅ Developed a modular workflow using JupyterLab, Python, and MLflow
- ✅ Applied Optuna for hyperparameter tuning across multiple regressors
- ✅ Implemented Ridge stacking and OOF ensembling for performance gains
- ✅ Tracked experiments and metrics using MLflow for reproducibility
- ✅ Practiced ML competition-style evaluation through leaderboard simulation

The final model demonstrated strong predictive performance and robustness on the test set. More importantly, the process reinforced production-grade ML practices, from data preprocessing to ensemble deployment.

This project helped me level up my skills not just in model training but in structuring, documenting, and presenting machine learning work like a real-world engineer.

---

## 💡 Next Steps

- Tune **Ridge alpha** and add to ensemble stack
- Visualize SHAP values for model interpretation
- Try greedy/hill climbing ensembles
- Package code into a CLI or deploy via Streamlit

---

## 🛠️ Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, MLflow, Matplotlib, Seaborn, Jupyter Notebook, Git, GitHub

---

## 🙋‍♂️ Author

**Saayed Alam**  
📌 [GitHub](https://github.com/saayedalam) | [Kaggle](https://www.kaggle.com/saayedalam) | [LinkedIn](https://www.linkedin.com/in/saayedalam)
