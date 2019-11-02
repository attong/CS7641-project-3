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
    rec_err_plot(xtrain2,18, "Random Projection Reconstruction Error dat2", "figs/rca/recondat2.png")
    rec_err_plot(xtrain1,54, "Random Projection Reconstruction Error dat1", "figs/rca/recondat1.png")

def inverse_transform(x,transformer):
    W = transformer.components_
    p = pinv(W)
    reconstructed = ((p@W)@(x.T)).T
    return reconstructed


def rec_err_plot(x, comps, title,filnam, reps=10):
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
    plt.ylabel("Reconstruction Error")
    ax.grid(which='major', linestyle='-', linewidth='1', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax.minorticks_on()
    meanerr = np.mean(error,axis=1)
    stderr = np.std(error,axis=1)
    print(np.mean(stderr))
    ax.plot(components,meanerr,'o-', color="r",label="mean reconstruction error",linewidth=4.0)
    plt.fill_between(components, meanerr - stderr,
                     meanerr + stderr, alpha=0.1,
                     color="r")
    plt.legend()
    plt.savefig(filnam)


if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()