import streamlit as st
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter , defaultdict
from itertools import combinations
import pandas as pd
from utils.text_processing import build_cooccurrence
import numpy as np

# Global theme
sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13
})


def plot_year_distribution(similar_df):

    st.subheader("Year-wise Distribution")

    year_counts = similar_df["year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8,5))

    sns.barplot(
        x=year_counts.index,
        y=year_counts.values,
        palette="coolwarm",
        ax=ax
    )

    ax.set_title("Distribution of Similar Cases by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Cases")

    st.pyplot(fig)


def plot_similarity(similar_df):

    similar_df = similar_df.sort_values("score", ascending=False)

    fig, ax = plt.subplots(figsize=(10,6))

    sns.barplot(
        x="score",
        y="file_name",
        data=similar_df,
        palette="coolwarm",
        ax=ax
    )

    ax.set_title("Top Similar Legal Cases")
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Case")

    ax.set_xlim(0,1)

    st.pyplot(fig)


def show_wordcloud(similar_df, df):

    text = ""

    for file in similar_df["file_name"]:
        text += df[df["file_name"] == file]["clean_text"].values[0]

    if text.strip() == "":
        return

    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        colormap="coolwarm"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10,5))

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)


def plot_top_legal_terms(vectorizer, embeddings, top_n=20):

    terms = vectorizer.get_feature_names_out()

    # sum tf-idf score for each word
    scores = np.asarray(embeddings.sum(axis=0)).flatten()

    # sort by score
    top_idx = scores.argsort()[::-1][:top_n]

    top_terms = [terms[i] for i in top_idx]
    top_scores = [scores[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10,6))

    sns.barplot(
        x=top_scores,
        y=top_terms,
        palette="mako",
        ax=ax
    )

    ax.set_title("Top Important Legal Terms (TF-IDF)")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Terms")

    st.pyplot(fig)

def plot_keyword_cooccurrence(cases_text_list, top_n=15):

    combined_cooc = defaultdict(int)

    # reuse existing function
    for text in cases_text_list:
        cooc = build_cooccurrence(text)

        for pair, count in cooc.items():
            combined_cooc[pair] += count

    # extract vocabulary
    vocab = set()
    for w1, w2 in combined_cooc.keys():
        vocab.add(w1)
        vocab.add(w2)

    vocab = list(vocab)[:top_n]

    # create matrix
    matrix = pd.DataFrame(0, index=vocab, columns=vocab)

    for (w1, w2), count in combined_cooc.items():
        if w1 in vocab and w2 in vocab:
            matrix.loc[w1, w2] += count
            matrix.loc[w2, w1] += count

    # plot heatmap
    fig, ax = plt.subplots(figsize=(10,8))

    sns.heatmap(
        matrix,
        cmap="coolwarm",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Legal Keyword Co-occurrence Heatmap")

    st.pyplot(fig)