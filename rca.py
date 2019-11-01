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


def main():
    #load normalized data
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    # kurt(xtrain1,55,"ICA Mean Kurtosis vs num_components dat1","figs/ica/kurtdat1.png")
    # kurt(xtrain2,20,"ICA Mean Kurtosis vs num_components dat2","figs/ica/kurtdat2.png")
    # rec_err_plot(xtrain2,18, "Reconstruction Error dat 2", "figs/ica/recon_err_dat2.png")
    # rec_err_plot(xtrain1,55, "Reconstruction Error dat 1", "figs/ica/recon_err_dat1.png")
    #dat1: 36, dat2: 3
    # rng = np.random.RandomState(42)
    # X = rng.rand(100, 10000)
    # transformer = GaussianRandomProjection(random_state=rng)
    # X_new = transformer.fit_transform(X)
    # print(X_new.shape)
    rec_err_plot(xtrain2,18, "Random Projection Reconstruction Error dat2")

def inverse_transform(x,transformer):
    W = transformer.components_
    p = pinv(W)
    reconstructed = ((p@W)@(x.T)).T
    return reconstructed


def rec_err_plot(x, comps, title, reps=10):
    components = np.arange(1,comps+1,1)
    err = []
    error= np.ones((comps,reps))
    fig, ax = plt.subplots()
    for i in range(reps):
        temp=[]
        for j in components:
            transformer = GaussianRandomProjection(n_components=j)
            transformer.fit(x)
            x_recon = inverse_transform(x , transformer)
            temp.append(reconstruction_error(x,x_recon))
            error[j-1,i]=reconstruction_error(x,x_recon)
        ax.plot(components, temp)
    plt.title(title)
    plt.xlabel("n_components")
    plt.ylabel("Mean Kurtosis")
    ax.grid(which='major', linestyle='-', linewidth='1', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax.minorticks_on()
    meanerr = np.mean(error,axis=1)
    stderr = np.mean(error,axis=1)
    print(meanerr)
    plt.show()


if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()