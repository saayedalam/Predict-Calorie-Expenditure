# src/utils.py

import os
import pandas as pd
from datetime import datetime

def log_score(model_name, score, notes="", path="../outputs/metrics.csv"):
    """Append model evaluation results to a CSV log."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": model_name,
        "score": round(score, 5),
        "notes": notes
    }

    df_entry = pd.DataFrame([entry])

    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        df_all = pd.concat([df_existing, df_entry], ignore_index=True)
    else:
        df_all = df_entry

    df_all.to_csv(path, index=False)
    print(f"✅ Logged: {model_name} | Score: {score:.5f}")
