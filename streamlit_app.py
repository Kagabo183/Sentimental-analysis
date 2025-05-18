import streamlit as st
import pandas as pd
from src.data_preprocessing import load_and_clean_data
from src.sentiment_analysis import analyze_sentiments
from src.visualization import (
    plot_sentiment_distribution,
    plot_top_comments,
    plot_compound_histogram,
    plot_wordclouds
)
import matplotlib.pyplot as plt

st.title("RURA Comments Sentiment Analysis")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess data
    df = load_and_clean_data(uploaded_file)
    df = analyze_sentiments(df)

    # Show data preview
    st.subheader("Data Preview")
    st.write(df.head())

    # Sentiment distribution plot
    st.subheader("Sentiment Distribution")
    fig1 = plt.figure()
    plot_sentiment_distribution(df)
    st.pyplot(fig1)

    # Top comments plot
    st.subheader("Top Positive and Negative Comments")
    fig2 = plt.figure(figsize=(16, 8))
    plot_top_comments(df)
    st.pyplot(fig2)

    # Compound score histogram
    st.subheader("Compound Score Distribution")
    fig3 = plt.figure()
    plot_compound_histogram(df)
    st.pyplot(fig3)

    # Word clouds
    st.subheader("Word Clouds for Positive and Negative Comments")
    fig4 = plt.figure(figsize=(12, 6))
    plot_wordclouds(df)
    st.pyplot(fig4)

else:
    st.info("Please upload a CSV file to begin.")
