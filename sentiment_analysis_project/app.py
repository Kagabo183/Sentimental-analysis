import streamlit as st
from src.data_preprocessing import load_and_clean_data
from src.sentiment_analysis import analyze_sentiments
from src.visualization import (
    plot_sentiment_distribution,
    plot_top_comments,
    plot_compound_histogram,
    plot_wordclouds
)

def main():
    st.title("Sentiment Analysis Dashboard")

    filepath = "C:/Users/RIZIKI/Downloads/RURA Comments Dataset.csv"  # Your CSV path
    df = load_and_clean_data(filepath)
    df = analyze_sentiments(df)

    st.header("Sentiment Distribution")
    fig1 = plot_sentiment_distribution(df)
    st.pyplot(fig1)

    st.header("Top Comments")
    fig2 = plot_top_comments(df)
    st.pyplot(fig2)

    st.header("Compound Score Histogram")
    fig3 = plot_compound_histogram(df)
    st.pyplot(fig3)

    st.header("Word Clouds")
    fig4 = plot_wordclouds(df)
    st.pyplot(fig4)

if __name__ == "__main__":
    main()
