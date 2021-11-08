import pandas as pd
import numpy as np
import re
import json
import sys
import os
import gzip
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

plt.style.use("fivethirtyeight")


def load_data(link, mode='r'):
    data = []
    with open(link, mode) as f:
        for line in f:
            data.append(json.loads(line.strip()))

    return pd.DataFrame.from_dict(data)


def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


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
    df = load_data()
    print(df)


if __name__ == '__main__':
    main()
