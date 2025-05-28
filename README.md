
# 🔥 A/B + Ensemble Project: Predicting Calorie Expenditure

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
/data/
    train.csv, test.csv
/notebooks/
    01_eda_feature_engineering.ipynb
    02_model_training_optuna.ipynb
    03_ensemble_and_stacking.ipynb
/src/
    models/
        train_model.py
        ensemble.py
    utils/
        preprocessing.py
        metrics.py
/outputs/
    submission_ensemble.csv
requirements.txt
README.md
```

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
