import pandas as pd
import numpy as np
import re
import json
import sys
import os
import gzip
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns


def load_data(link, mode='r'):
    print("Loading data...")
    data = []
    with open(link, mode) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print("Done!")
    return pd.DataFrame.from_dict(data)


def clean_data(df):
    print("Cleaning data...")
    df = df.loc[(df['verified'])]
    df = df.drop(columns=['image', 'style', 'unixReviewTime', 'verified', 'summary'])
    for col in df.columns:
        if np.any((df.isna()[col])):
            df[col] = df[col].fillna("" if type(df[col][0]) is str else 0)
    print(f"Remaining columns are {[x for x in df.columns]}")
    print("Done!")
    return df


def get_word_cloud(data, sample=100000):
    print("Creating word cloud plot...")
    combined = ' '.join([txt for txt in np.random.choice(data, min(sample, len(data)))])
    word_cloud = WordCloud(width=800, height=600, random_state=21, max_font_size=150).generate(combined)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('graphs/wordcloud.png')
    print("Image saved as wordcloud.png")
    print("Done!")


def visualize(df, sample=1000, method='total'):
    print("Creating scatter plots...")
    plt.figure(figsize=(8, 6))
    df.sample(min(sample, df.shape[0])).plot.scatter(x='polar', y='sub', c='overall', colormap='viridis')
    plt.title("Polarity vs Subjectivity")
    plt.xlabel("Polarity Score")
    plt.ylabel("Subjectivity Score")
    plt.grid(True)
    plt.savefig('graphs/PolarSubScatter.png.png')
    print("Image 1/4 saved as PolarSubScatter.png")

    plt.figure(figsize=(8, 6))
    df.sample(min(sample, df.shape[0])).plot.scatter(x='overall', y=method, c='overall', colormap='viridis')
    plt.title("Rating vs Sentiment")
    plt.xlabel("Overall Rating")
    plt.ylabel("Sentiment Score")
    plt.grid(True)
    plt.savefig('graphs/RatingSentimentScatter.png')
    print("Image 2/4 saved as RatingSentimentScatter.png")

    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots()
    plt.boxplot([df[df['overall'] == i][method] for i in [1.0, 2.0, 3.0, 4.0, 5.0]],
                patch_artist=True,
                boxprops=dict(facecolor='yellow', color='purple'),
                capprops=dict(color='purple'),
                whiskerprops=dict(color='purple'),
                flierprops=dict(color='purple', markeredgecolor='purple'),
                medianprops=dict(color='red'))
    ax.set_xticklabels(['1.0', '2.0', '3.0', '4.0', '5.0'])
    plt.title("Rating vs Sentiment")
    plt.xlabel("Overall Rating")
    plt.ylabel("Sentiment Score")
    plt.savefig('graphs/RatingSentimentBoxplot.png')
    print("Image 3/4 saved as RatingSentimentBoxplot.png")

    plt.figure(figsize=(8, 6))
    plt.hist(df[df['overall'] == 1.0][method], alpha=0.3, label='1.0', bins=50)
    plt.hist(df[df['overall'] == 2.0][method], alpha=0.3, label='2.0', bins=50)
    plt.hist(df[df['overall'] == 3.0][method], alpha=0.3, label='3.0', bins=50)
    plt.hist(df[df['overall'] == 4.0][method], alpha=0.3, label='4.0', bins=50)
    plt.hist(df[df['overall'] == 5.0][method], alpha=0.3, label='5.0', bins=50)
    plt.legend()
    plt.title("Sentiment histogram")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.savefig('graphs/SentimentHist.png')
    print("Image 4/4 saved as SentimentHist.png")
    print("Done!")


def get_polarity(text):
    return TextBlob(text).sentiment.polarity


def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def main():
    df = load_data('data/Video_Games.json')
    df = clean_data(df)
    get_word_cloud(df['reviewText'])

    print("Performing sentiment analysis...")
    dfs = df.sample(min(100000, df.shape[0]))
    dfs['polar'] = dfs['reviewText'].apply(get_polarity)
    dfs['sub'] = dfs['reviewText'].apply(get_subjectivity)
    dfs['total'] = dfs['polar'] + dfs['sub']

    dfs = dfs.sort_values('total', ascending=False)
    # print(dfs['total'].values[:3], dfs['total'].values[-3:])
    print(f"""Three most positive review: 
            {dfs['reviewText'].values[0]}
            {dfs['reviewText'].values[1]}
            {dfs['reviewText'].values[2]}""")
    print(f"""Three most negative review: 
            {dfs['reviewText'].values[-1]}
            {dfs['reviewText'].values[-2]}
            {dfs['reviewText'].values[-3]}""")
    print()

    dfss = dfs.drop(columns=['reviewTime', 'asin', 'reviewText', 'reviewerID', 'polar', 'sub'])
    dfgb = dfss.groupby(['reviewerName']).sum()
    dfgb = dfgb.sort_values('total', ascending=False)
    print(f"Three most positive reviewer: {[x for x in dfgb.index[:3]]}")
    print(f"Three most negative reviewer: {[x for x in dfgb.index[-3:]]}")
    print("Done!")

    visualize(dfs, method='polar')

    print("All modules finished!!!")


if __name__ == '__main__':
    main()
