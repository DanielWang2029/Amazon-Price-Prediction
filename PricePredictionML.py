import pandas as pd
import numpy as np
import re
import json
import sys
import os
import gzip
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data(link, mode='r'):
    print("Loading data...")
    df = pd.read_json(link)
    print("Done!")
    return df


def clean_data(df):
    print("Cleaning data...")
    df = df.drop(columns=['category', 'description', 'title'])
    df = df.dropna()
    df = df.loc[(df['main_cat'] != '') & (df['main_cat'].apply(lambda x: '_' not in x))]
    # df = df.loc[(df['feature'].apply(lambda x: len(x) != 0))]
    df = df.loc[(df['also_buy'].apply(lambda x: len(x) != 0))]
    df = df.loc[(df['also_view'].apply(lambda x: len(x) != 0))]
    df = df.loc[(df['rank'].apply(lambda x: len(x) >= 2))]
    df = df.loc[(df['rank'].apply(lambda x: len(re.findall('[1-9][0-9]+', x[0].replace(',', ''))) > 0))]
    df = df.loc[(df['rank'].apply(lambda x: len(re.findall('[1-9][0-9]+', x[1].replace(',', ''))) > 0))]
    print(f"Remaining columns are {[(x, type(df[x].values[0])) for x in df.columns]}")
    print("Done!")
    return df


def process_data(df, method=np.median):
    print("Preprocessing data...")

    df['feature_count'] = df['feature'].apply(lambda x: len(x))

    # create main_cat_val column
    dic = {}
    print(np.unique(df['main_cat'].values))
    for mt in np.unique(df['main_cat'].values):
        dic[mt] = method(df.loc[df['main_cat'] == mt]['price'].values)
    df['main_cat_val'] = [dic[x] for x in df['main_cat']]

    # create overall_rank_val and specific_rank_val columns
    df['rank'] = df['rank'].apply(lambda x: x[:2])
    df['rank'] = df['rank'].apply(lambda x: [int(re.findall('[1-9][0-9]+', y.replace(',', ''))[0]) for y in x])
    df['overall_rank_val'] = df['rank'].apply(lambda x: 1 / x[0])
    df['specific_rank_val'] = df['rank'].apply(lambda x: 1 / x[1])
    print("Done!")
    return df


class Net(nn.Module):

    def __init__(self, input_size=5, hidden_size=3):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def main():
    df = load_data('data/cleanedAppliances.json')
    df = clean_data(df)
    df = process_data(df)
    print(df.head())
    print(df.shape)


if __name__ == '__main__':
    main()