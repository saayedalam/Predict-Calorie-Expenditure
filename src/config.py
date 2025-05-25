# src/config.py

# Core libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error

# Tree-based models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Optimization
import optuna

# Global plot settings
sns.set(style="whitegrid")
#plt.rcParams["figure.figsize"] = (10, 6)

# Optional: suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Paths (you can expand this later)
DATA_PATH = "/home/saayed/data-portfolio/kaggle-1/data/"
