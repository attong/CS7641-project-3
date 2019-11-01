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
from sklearn.decomposition import PCA

def eigenratioplot(x,title, filnam):
    plt.clf()
    pca = PCA()
    pca.fit(x)
    plt.plot(np.arange(1,len(pca.explained_variance_ratio_)+1,1),np.cumsum(pca.explained_variance_ratio_),marker='o')
    plt.ylabel("Cum Sum Explained Variance Ratio")
    plt.xlabel("Components")
    plt.title(title)
    plt.savefig(filnam)
    return

def reconstruction_error(original,transformed):
    return ((original - transformed) ** 2).mean()

def rec_err_plot(x, comps, title, filnam):
    components = np.arange(1,comps+1,1)
    err = []
    for c in components:
        temp=[]
        for i in range(5):
            pca= PCA(n_components=c)
            dataset = pca.fit_transform(x)
            transformed = pca.inverse_transform(dataset)
            temp.append(reconstruction_error(x,transformed))
        err.append(float(sum(temp))/len(temp))
    plt.clf()
    plt.plot(components, err,marker='o')
    plt.xlabel("Components")
    plt.ylabel("Reconstruction Error")
    plt.title(title)
    plt.grid(b=True)
    plt.savefig(filnam)
    return

def k(x,comp):
    pca = PCA(n_components=comp)
    data = pca.fit_transform(x)
    elbowplot(data,20,"silhouette", "tempdat2","figs/pca/temp.png",elbow=False)

def main():
    #load normalized data
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    print(xtrain1.shape)
    # eigenratioplot(xtrain2, "Cum Sum Explained Variance Ratio per Component dat2", "figs/pca/explainedvar_dat2.png")
    # eigenratioplot(xtrain1, "Cum Sum Explained Variance Ratio per Component dat1", "figs/pca/explainedvar_dat1.png")
    # rec_err_plot(xtrain2,18, "Reconstruction Error dat 2", "figs/pca/recon_err_dat2.png")
    # rec_err_plot(xtrain1,54, "Reconstruction Error dat 1", "figs/pca/recon_err_dat1.png")
    k(xtrain2,8)


"""
pca=PCA(n_components=5)
dataset = pca.fit_transform(data)
pca.inverse_transform(X_train_pca)
"""


if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()