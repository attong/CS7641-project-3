#Independent component analysis
from app import load_data
import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from kmeans import elbowplot
from sklearn.decomposition import FastICA
from pca import reconstruction_error


def main():
    #load normalized data
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    # kurt(xtrain1,55,"ICA Mean Kurtosis vs num_components dat1","figs/ica/kurtdat1.png")
    # kurt(xtrain2,20,"ICA Mean Kurtosis vs num_components dat2","figs/ica/kurtdat2.png")
    rec_err_plot(xtrain2,18, "Reconstruction Error dat 2", "figs/ica/recon_err_dat2.png")
    rec_err_plot(xtrain1,55, "Reconstruction Error dat 1", "figs/ica/recon_err_dat1.png")
    #dat1: 36, dat2: 3

def rec_err_plot(x, comps, title, filnam):
    components = np.arange(1,comps+1,1)
    err = []
    for c in components:
        temp=[]
        for i in range(5):
            ica= FastICA(n_components=c, max_iter = 500, tol=0.01)
            dataset = ica.fit_transform(x)
            transformed = ica.inverse_transform(dataset)
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

def kurt(x,d,title,filnam):
    plt.clf()
    dimensions = np.arange(1,d+1,1)
    kurt=[]
    for i in dimensions:
        ica=FastICA(n_components=i, max_iter = 500, tol=0.01)
        dataset=ica.fit_transform(x)
        temp = pd.DataFrame(dataset)
        k = temp.kurt(axis=0)
        kurt.append(k.abs().mean())
        # print(k)
    fig, ax = plt.subplots()
    ax.plot(dimensions,kurt)
    plt.title(title)
    plt.xlabel("n_components")
    plt.ylabel("Mean Kurtosis")
    ax.grid(which='major', linestyle='-', linewidth='1', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax.minorticks_on()
    plt.xticks(np.arange(0,d+1,5))
    plt.savefig(filnam)


if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()