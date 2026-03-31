import pandas as pd
import datetime
import json
import os
from config import HISTORY_FILE


def save_to_history(similar_df, filename, keywords=""):

    avg_score = float(similar_df["score"].mean())
    top_case = similar_df.iloc[0]["file_name"]

    year_distribution = similar_df["year"].value_counts().to_dict()

    record = {
        "file_name": filename,
        "timestamp": str(datetime.datetime.now()),
        "avg_score": avg_score,
        "top_case": top_case,
        "keywords": keywords,
        "year_distribution": json.dumps(year_distribution),
        "results": json.dumps(similar_df.to_dict())
    }

    new_df = pd.DataFrame([record])

    if os.path.exists(HISTORY_FILE):
        old_df = pd.read_excel(HISTORY_FILE)
        updated_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        updated_df = new_df

    updated_df.to_excel(HISTORY_FILE, index=False)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame()
    return pd.read_excel(HISTORY_FILE)