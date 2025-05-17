from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def analyze_sentiments(df):
    vader = SentimentIntensityAnalyzer()
    df["Sentiment Scores"] = df["Translated Comments (English)"].apply(vader.polarity_scores)
    df["Compound"] = df["Sentiment Scores"].apply(lambda x: x["compound"])
    df["Sentiment Label"] = df["Compound"].apply(
        lambda score: "Positive" if score > 0.05 else ("Negative" if score < -0.05 else "Neutral")
    )
    return df
