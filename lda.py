#principal component analysis
from app import load_data
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from kmeans import elbowplot
from sklearn.decomposition import FastICA
from pca import reconstruction_error
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from scipy.linalg import pinv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
    #load normalized data
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    varianceratioplot(xtrain2,ytrain2,"LDA Cum Sum Variance dat2","figs/lda/varianceratiodat2.png")

def varianceratioplot(x,y,title, filnam):
    plt.clf()
    pca = LinearDiscriminantAnalysis()
    pca.fit(x,y)
    plt.plot(np.arange(1,len(pca.explained_variance_ratio_)+1,1),np.cumsum(pca.explained_variance_ratio_),marker='o')
    plt.ylabel("Cum Sum Explained Variance Ratio")
    plt.xlabel("Components")
    plt.title(title)
    plt.savefig(filnam)
    return



if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()