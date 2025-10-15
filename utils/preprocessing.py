import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])


    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])


    X = df.drop('Class', axis=1).values
    y = df['Class'].values


    return X, y, df


def plot_pca(X, y, title='PCA Visualization'):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(reduced[y==0,0], reduced[y==0,1], label='Normal', alpha=0.5)
    plt.scatter(reduced[y==1,0], reduced[y==1,1], label='Fraud', alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.show()