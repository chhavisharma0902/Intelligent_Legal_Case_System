import streamlit as st
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.pdf_reader import extract_text_from_pdf
from utils.text_processing import clean_text, extract_keywords
from utils.similarity import find_similar_cases, search_by_keywords
from utils.analytics import plot_year_distribution, plot_similarity, show_wordcloud , plot_top_legal_terms,plot_keyword_cooccurrence
from utils.history_manager import save_to_history, load_history
from config import DATASET_FILE



# --------------------------------
# LOAD DATASET
# --------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_excel(DATASET_FILE)
    df["clean_text"] = df["clean_text"].fillna("")
    return df


# --------------------------------
# TF-IDF MODEL
# --------------------------------
@st.cache_resource
def get_vectorizer_and_embeddings(df):

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )

    embeddings = vectorizer.fit_transform(df["clean_text"])

    return vectorizer, embeddings



# --------------------------------
# UI TITLE
# --------------------------------
st.title("⚖️ Intelligent Legal Case Similarity System")

menu = st.sidebar.selectbox(
    "Menu",
    ["Search Case", "View History"]
)


# ============================================
# SEARCH PAGE
# ============================================
if menu == "Search Case":

    st.header("Search Similar Legal Cases")

    input_mode = st.radio(
        "Choose Input Method",
        ["Upload PDF", "Enter Text", "Keyword Search"]
    )

    text = None
    filename = "Manual_Input"

    # ------------------------------
    # PDF INPUT
    # ------------------------------
    if input_mode == "Upload PDF":

        uploaded_file = st.file_uploader(
            "Upload your legal case PDF",
            type=["pdf"]
        )

        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
            filename = uploaded_file.name

    # ------------------------------
    # KEYWORD SEARCH
    # ------------------------------
    elif input_mode == "Keyword Search":

        keywords = st.text_input(
            "Enter keywords:",
            placeholder="example: freedom speech constitution amendment"
        )

        if keywords:

            with st.spinner("Searching legal database..."):

                df = load_dataset()
                vectorizer, embeddings = get_vectorizer_and_embeddings(df)

                results_df = search_by_keywords(
                    keywords,
                    df,
                    vectorizer,
                    embeddings
                )

            st.subheader("Top Cases for Keywords")
            st.dataframe(results_df)

            # SAVE HISTORY (IMPORTANT FIX)
            save_to_history(results_df, "Keyword_Search", keywords)

            cases_text_list_keyword = []

            for file in results_df["file_name"]:
                text = df[df["file_name"] == file]["clean_text"].values[0]
                cases_text_list_keyword.append(text)

            subset_embeddings_keyword = vectorizer.transform(cases_text_list_keyword)

            st.success("Search saved to history")
            st.header("Analytics")

            plot_year_distribution(results_df)
            plot_similarity(results_df)
            plot_top_legal_terms(vectorizer ,subset_embeddings_keyword)
            if st.checkbox("Show WordCloud"):
                show_wordcloud(vectorizer,subset_embeddings_keyword)

    # ------------------------------
    # TEXT INPUT
    # ------------------------------
    else:

        text = st.text_area(
            "Enter legal case text:",
            height=250,
            placeholder="Paste legal case text here..."
        )

    # ------------------------------
    # PROCESS TEXT / PDF
    # ------------------------------
    if text and input_mode != "Keyword Search":

        with st.spinner(" Analyzing case..."):

            cleaned = clean_text(text)

            st.subheader("Input Preview")
            st.write(cleaned[:300])

            df = load_dataset()

            vectorizer, embeddings = get_vectorizer_and_embeddings(df)

            similar_df = find_similar_cases(
                cleaned,
                df,
                vectorizer,
                embeddings
            )

        # ------------------------------
        # RESULTS
        # ------------------------------
        st.subheader("Top 5 Similar Cases")

        st.dataframe(similar_df)

        # EXTRACT KEYWORDS
        keywords = extract_keywords(cleaned)

        # SAVE HISTORY
        save_to_history(similar_df, filename, keywords)

        st.success("Search saved to history")
        cases_text_list = []

        for file in similar_df["file_name"]:
            text = df[df["file_name"] == file]["clean_text"].values[0]
            cases_text_list.append(text)

        subset_embeddings = vectorizer.transform(cases_text_list)
        # ------------------------------
        # ANALYTICS
        # ------------------------------
        st.header("Analytics")

        plot_year_distribution(similar_df)
        plot_top_legal_terms(vectorizer,subset_embeddings)
        plot_similarity(similar_df)
        plot_keyword_cooccurrence(vectorizer,subset_embeddings,cases_text_list)
        st.subheader("Important Legal Terms WordCloud")
        show_wordcloud(vectorizer,subset_embeddings)


# ============================================
# HISTORY PAGE
# ============================================
elif menu == "View History":

    st.header("Search History")

    history_df = load_history()

    if history_df.empty:
        st.write("No history available.")

    else:

        # FIX NaN keywords
        if "keywords" in history_df.columns:
            history_df["keywords"] = history_df["keywords"].fillna("")
        else:
            history_df["keywords"] = ""

        for _, row in history_df.iterrows():

            st.subheader(f"📄 {row['file_name']}")

            st.write("🔑 Keywords:", row["keywords"])

            st.write("🕒 Time:", row["timestamp"])

            st.write("⭐ Avg Similarity:", round(row["avg_score"], 3))

            st.write("🏆 Top Match:", row["top_case"])

            # ------------------------------
            # YEAR DISTRIBUTION
            # ------------------------------
            if pd.notna(row["year_distribution"]):
                year_dist = json.loads(row["year_distribution"])
            else:
                year_dist = {}

            st.write("📊 Year Distribution")

            st.bar_chart(year_dist)

            # ------------------------------
            # RESULTS TABLE
            # ------------------------------
            results = pd.DataFrame(
                json.loads(row["results"])
            )

            st.dataframe(results)

            st.markdown("---")

