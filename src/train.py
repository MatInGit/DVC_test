import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-D","--data", default="data/IRIS.csv")
parser.add_argument("-O","--out", default="models/reg_model.pkl")
args = parser.parse_args()

import yaml

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)

seed = params['train']['seed']
split = params['train']['split']


def main():
    
    df = pd.read_csv(args.data)

    X = df.loc[:, df.columns != 'species']
    y = df.loc[:, df.columns == 'species']
    y = pd.factorize(df['species'])[0] + 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    reg = LinearRegression().fit(X, y)
    print("acc:",reg.score(X_train, y_train))

    pickle.dump(reg, open(args.out, 'wb'))


if __name__ == '__main__':
    main()