import sqlite3
import pandas as pd
import datetime
import json
from config import HISTORY_DB


def get_connection():
    return sqlite3.connect(HISTORY_DB)


def initialize_db():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            timestamp TEXT,
            avg_score REAL,
            top_case TEXT,
            keywords TEXT,
            year_distribution TEXT,
            results TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_to_history(similar_df, filename, keywords=""):

    initialize_db()

    avg_score = float(similar_df["score"].mean())
    top_case = similar_df.iloc[0]["file_name"]

    year_distribution = similar_df["year"].value_counts().to_dict()

    record = (
        filename,
        str(datetime.datetime.now()),
        avg_score,
        top_case,
        str(keywords),
        json.dumps(year_distribution),
        json.dumps(similar_df.to_dict())
    )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history
        (file_name, timestamp, avg_score, top_case, keywords, year_distribution, results)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, record)

    conn.commit()
    conn.close()


def load_history():

    initialize_db()

    conn = get_connection()

    df = pd.read_sql_query(
        "SELECT * FROM history ORDER BY id DESC",
        conn
    )

    conn.close()

    return df