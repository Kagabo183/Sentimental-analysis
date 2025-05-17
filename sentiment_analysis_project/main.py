from src.data_preprocessing import load_and_clean_data
from src.sentiment_analysis import analyze_sentiments
from src.visualization import (
    plot_sentiment_distribution,
    plot_top_comments,
    plot_compound_histogram,
    plot_wordclouds
)

def main():
    filepath = "C:/Users/RIZIKI/Downloads/RURA Comments Dataset.csv"  # Use your dataset absolute path here
    df = load_and_clean_data(filepath)
    df = analyze_sentiments(df)
    
    plot_sentiment_distribution(df)
    plot_top_comments(df)
    plot_compound_histogram(df)
    plot_wordclouds(df)

if __name__ == "__main__":
    main()
