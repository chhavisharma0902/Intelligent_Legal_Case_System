Intelligent Legal Case Analysis System
Overview

This project is an NLP-based system that analyzes legal case documents and identifies similar judgments. It helps users explore relationships between cases by applying text similarity and linguistic analysis techniques.

The system is built using Python and deployed as an interactive web application using Streamlit.

Dataset

The dataset used in this project is:

Legal Dataset: SC Judgments India (1950–2024)

Source: Kaggle
Collected from Indian Kanoon
Contains Supreme Court judgments with over 98% coverage
Key Concepts Used

The system applies several Natural Language Processing techniques:

Text preprocessing and cleaning
TF-IDF vectorization
Cosine similarity for case comparison
Named Entity Recognition (NER)
Keyword co-occurrence matrix analysis

These methods help identify meaningful relationships between legal documents.

Features
Upload and analyze legal case documents
Find similar judgments using NLP similarity techniques
Visualize important legal terms and relationships
Interactive analytics dashboard with plots and word clouds
Keyword co-occurrence heatmaps for legal insights

Technologies Used
Python
Streamlit – web application framework
Pandas – data handling
scikit-learn – TF-IDF and similarity computation
spaCy – Named Entity Recognition
Seaborn and Matplotlib – data visualization

How to Run the Project
Clone the repository
Install dependencies
  pip install -r requirements.txt
Run the application
  streamlit run app.py

Purpose of the Project

The goal of this project is to explore how Natural Language Processing can assist in analyzing legal documents and identifying patterns across judgments. It demonstrates how machine learning and text analytics can be applied to real-world legal datasets.
