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

plt.style.use("fivethirtyeight")


def load_data(link, mode='r'):
    print("Loading data...")
    data = []
    with open(link, mode) as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return pd.DataFrame.from_dict(data)


def clean_data(df):
    print("Cleaning data...")
    df = df.loc[(df['verified'])]
    df = df.drop(columns=['image', 'style', 'reviewerName', 'unixReviewTime', 'verified'])
    for col in df.columns:
        if np.any((df.isna()[col])):
            df[col] = df[col].fillna("" if type(df[col][0]) is str else 0)
    return df


def get_polarity(text):
    return TextBlob(text).sentiment.polarity


def analyze(rating):
    if rating > 0:
        return 'positive'
    elif rating < 0:
        return 'negative'
    else:
        return 'zero'


def main():
    df = load_data('data/Video_Games.json')
    df = clean_data(df)
    print(df.columns)


if __name__ == '__main__':
    main()
