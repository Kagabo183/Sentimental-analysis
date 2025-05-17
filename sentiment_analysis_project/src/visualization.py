import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from wordcloud import WordCloud

# Set seaborn style
sns.set(style="whitegrid")

def plot_sentiment_distribution(df):
    custom_palette = {
        "Negative": "#f08080",  # Light red
        "Neutral": "#f7e967",   # Soft yellow
        "Positive": "#90ee90"   # Light green
    }

    plt.figure(figsize=(7, 5))
    # Fix FutureWarning by using 'hue' or removing palette or adjusting call
    # Since you want countplot by x only, remove palette and set colors manually:
    order = ["Negative", "Neutral", "Positive"]
    counts = df["Sentiment Label"].value_counts().reindex(order)
    colors = [custom_palette[label] for label in order]

    ax = sns.barplot(x=counts.index, y=counts.values, palette=colors)

    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        percentage = 100 * count / total
        ax.annotate(f'{count}\n({percentage:.1f}%)',
                    (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                    ha='center', va='bottom', fontsize=10, color='black')

    plt.title("Sentiment Distribution of Comments", fontsize=14, pad=20)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Number of Comments", fontsize=12)
    plt.tight_layout(pad=3)
    plt.show()

def plot_top_comments(df):
    top_positive = df.sort_values("Compound", ascending=False).head(5)
    top_negative = df.sort_values("Compound", ascending=True).head(5)

    # Use pd.concat instead of deprecated append()
    top_combined = pd.concat([top_negative, top_positive])
    top_combined['Sentiment'] = ['Negative'] * 5 + ['Positive'] * 5

    def wrap_text(text, width=90):
        return "\n".join(textwrap.wrap(text, width=width))

    top_combined["Wrapped Comments"] = top_combined["Translated Comments (English)"].apply(
        lambda x: wrap_text(x, width=90)
    )

    plt.figure(figsize=(16, 8))
    sns.barplot(
        x="Compound",
        y="Wrapped Comments",
        hue="Sentiment",
        data=top_combined,
        palette={"Positive": "green", "Negative": "red"}
    )

    plt.title("Top 5 Most Positive and Negative Comments")
    plt.xlabel("Compound Sentiment Score")
    plt.ylabel("Comment")
    plt.legend(title="Sentiment")
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

def plot_compound_histogram(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Compound"], bins=20, kde=True, color='skyblue')
    plt.title("Distribution of Sentiment Compound Scores")
    plt.xlabel("Compound Sentiment Score")
    plt.ylabel("Number of Comments")
    plt.axvline(0, color='red', linestyle='--', label='Neutral Boundary')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_wordclouds(df):
    positive_text = " ".join(df[df["Sentiment Label"] == "Positive"]["Translated Comments (English)"])
    negative_text = " ".join(df[df["Sentiment Label"] == "Negative"]["Translated Comments (English)"])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    wordcloud_pos = WordCloud(width=600, height=400, background_color='white', colormap='Greens').generate(positive_text)
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title("Positive Comments Word Cloud")

    plt.subplot(1, 2, 2)
    wordcloud_neg = WordCloud(width=600, height=400, background_color='white', colormap='Reds').generate(negative_text)
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title("Negative Comments Word Cloud")

    plt.tight_layout()
    plt.show()
