import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from wordcloud import WordCloud

sns.set(style="whitegrid")

def plot_sentiment_distribution(df):
    custom_palette = {
        "Negative": "#f08080",  # Light red
        "Neutral": "#f7e967",   # Soft yellow
        "Positive": "#90ee90"   # Light green
    }

    order = ["Negative", "Neutral", "Positive"]
    counts = df["Sentiment Label"].value_counts().reindex(order)
    colors = [custom_palette[label] for label in order]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, palette=colors, ax=ax)

    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        percentage = 100 * count / total
        ax.annotate(f'{count}\n({percentage:.1f}%)',
                    (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                    ha='center', va='bottom', fontsize=10, color='black')

    ax.set_title("Sentiment Distribution of Comments", fontsize=14, pad=20)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Number of Comments", fontsize=12)
    fig.tight_layout(pad=3)

    plt.close(fig)  # Prevent double rendering in some environments
    return fig


def plot_top_comments(df):
    top_positive = df.sort_values("Compound", ascending=False).head(5)
    top_negative = df.sort_values("Compound", ascending=True).head(5)

    top_combined = pd.concat([top_negative, top_positive])
    top_combined['Sentiment'] = ['Negative'] * 5 + ['Positive'] * 5

    def wrap_text(text, width=90):
        return "\n".join(textwrap.wrap(text, width=width))

    top_combined["Wrapped Comments"] = top_combined["Translated Comments (English)"].apply(
        lambda x: wrap_text(x, width=90)
    )

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(
        x="Compound",
        y="Wrapped Comments",
        hue="Sentiment",
        data=top_combined,
        palette={"Positive": "green", "Negative": "red"},
        ax=ax
    )

    ax.set_title("Top 5 Most Positive and Negative Comments")
    ax.set_xlabel("Compound Sentiment Score")
    ax.set_ylabel("Comment")
    ax.legend(title="Sentiment")
    fig.tight_layout(rect=[0, 0, 1, 1])

    plt.close(fig)
    return fig


def plot_compound_histogram(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Compound"], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title("Distribution of Sentiment Compound Scores")
    ax.set_xlabel("Compound Sentiment Score")
    ax.set_ylabel("Number of Comments")
    ax.axvline(0, color='red', linestyle='--', label='Neutral Boundary')
    ax.legend()
    fig.tight_layout()

    plt.close(fig)
    return fig


def plot_wordclouds(df):
    positive_text = " ".join(df[df["Sentiment Label"] == "Positive"]["Translated Comments (English)"])
    negative_text = " ".join(df[df["Sentiment Label"] == "Negative"]["Translated Comments (English)"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    wordcloud_pos = WordCloud(width=600, height=400, background_color='white', colormap='Greens').generate(positive_text)
    axes[0].imshow(wordcloud_pos, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title("Positive Comments Word Cloud")

    wordcloud_neg = WordCloud(width=600, height=400, background_color='white', colormap='Reds').generate(negative_text)
    axes[1].imshow(wordcloud_neg, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title("Negative Comments Word Cloud")

    fig.tight_layout()

    plt.close(fig)
    return fig
