#principal component analysis
from app import load_data
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.decomposition import PCA

def eigenratioplot(x):
    pca = PCA()
    pca.fit(x)
    plt.plot(np.arange(1,len(pca.explained_variance_ratio_)+1,1),np.cumsum(pca.explained_variance_ratio_))
    plt.show()

def main():
    #load normalized data
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    pca= PCA()
    pca.fit(xtrain2)
    print(pca.explained_variance_ratio_)
    eigenratioplot(xtrain1)
"""
pca=PCA(n_componenets=5)
dataset = pca.fit_transform(data)
"""


if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()