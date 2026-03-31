import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_processing import cooccurrence_score

def find_similar_cases(query_text, df, vectorizer, embeddings):

    query_vec = vectorizer.transform([query_text])
    tfidf_scores = cosine_similarity(query_vec, embeddings).flatten()

    final_scores = []

    for i in range(len(df)):
        cooc_score = cooccurrence_score(query_text, df.iloc[i]["clean_text"])
        cooc_score = cooc_score / (cooc_score + 1)

        score = (0.7 * tfidf_scores[i]) + (0.3 * cooc_score)
        final_scores.append(score)

    final_scores = np.array(final_scores)

    top_idx = np.argsort(final_scores)[-5:][::-1]

    results = []
    for i in top_idx:
        results.append({
            "file_name": df.iloc[i]["file_name"],
            "year": df.iloc[i]["year"],
            "score": round(final_scores[i], 3),
            "preview": " ".join(df.iloc[i]["clean_text"].split()[:30])
        })

    return pd.DataFrame(results)

def search_by_keywords(query, df, vectorizer, embeddings):

    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, embeddings).flatten()

    top_idx = scores.argsort()[-5:][::-1]

    results = []

    for i in top_idx:

        results.append({
                "file_name": df.iloc[i]["file_name"],
                "year": df.iloc[i]["year"],
                "score": round(scores[i],3),
                "preview": " ".join(df.iloc[i]["clean_text"].split()[:30])
        })

    return pd.DataFrame(results)

