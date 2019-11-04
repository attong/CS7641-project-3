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
from sklearn.mixture import GaussianMixture
import time


def main():
    #load normalized data
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    kurt(xtrain1,55,"ICA Mean Kurtosis vs num_components dat1","figs/ica/kurtdat1.png")
    kurt(xtrain2,20,"ICA Mean Kurtosis vs num_components dat2","figs/ica/kurtdat2.png")
    rec_err_plot(xtrain2,18, "Reconstruction Error dat 2", "figs/ica/recon_err_dat2.png")
    rec_err_plot(xtrain1,55, "Reconstruction Error dat 1", "figs/ica/recon_err_dat1.png")
    ica = FastICA(n_components=3)
    data=ica.fit_transform(xtrain2)
    vary_k(xtrain2,data, 20, ytrain2, "dat2test")
    ica = FastICA(n_components=36)
    data=ica.fit_transform(xtrain1)
    vary_k(xtrain1,data, 50, ytrain1, "dat1")
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

def vary_k(x, transformedx, maxk,y,datnam,iters=5):
    scores=[]
    rang = np.arange(2,maxk,1)
    #scores for untransformed data
    sil_scores = np.ones((iters,maxk-2))
    arand_scores = np.ones((iters,maxk-2))
    ami_scores = np.ones((iters,maxk-2))
    km_sil_scores = np.ones((iters,maxk-2))
    km_arand_scores = np.ones((iters,maxk-2))
    km_ami_scores = np.ones((iters,maxk-2))
    em_time=np.ones((iters,maxk-2))
    km_time = np.ones((iters,maxk-2)) 

    #scores for transformed data
    transformed_sil_scores = np.ones((iters,maxk-2))
    transformed_arand_scores = np.ones((iters,maxk-2))
    transformed_ami_scores = np.ones((iters,maxk-2))
    transformed_km_sil_scores = np.ones((iters,maxk-2))
    transformed_km_arand_scores = np.ones((iters,maxk-2))
    transformed_km_ami_scores = np.ones((iters,maxk-2))
    transformed_em_time=np.ones((iters,maxk-2))
    transformed_km_time = np.ones((iters,maxk-2))

    for i in range(iters):
        for k in rang:
            #original data
            gm=GaussianMixture(n_components=k)
            start=time.time()
            clusters = gm.fit_predict(x)
            time1=time.time()
            em_time[i][k-2]=time1-start
            sil_scores[i][k-2]=metrics.silhouette_score(x,clusters)
            arand_scores[i][k-2]=metrics.adjusted_rand_score(y,clusters)
            ami_scores[i][k-2]=metrics.adjusted_mutual_info_score(y,clusters)

            km=KMeans(n_clusters=k)
            start= time.time()
            km_clusters=km.fit_predict(x)
            km_time[i][k-2]=time.time()-start
            km_sil_scores[i][k-2]=metrics.silhouette_score(x,km_clusters)
            km_arand_scores[i][k-2]=metrics.adjusted_rand_score(y,km_clusters)
            km_ami_scores[i][k-2]=metrics.adjusted_mutual_info_score(y,km_clusters)

            #transformed data
            gm=GaussianMixture(n_components=k)
            start=time.time()
            clusters = gm.fit_predict(transformedx)
            time1=time.time()
            transformed_em_time[i][k-2]=time1-start
            transformed_sil_scores[i][k-2]=metrics.silhouette_score(transformedx,clusters)
            transformed_arand_scores[i][k-2]=metrics.adjusted_rand_score(y,clusters)
            transformed_ami_scores[i][k-2]=metrics.adjusted_mutual_info_score(y,clusters)

            km=KMeans(n_clusters=k)
            start= time.time()
            km_clusters=km.fit_predict(transformedx)
            transformed_km_time[i][k-2]=time.time()-start
            transformed_km_sil_scores[i][k-2]=metrics.silhouette_score(transformedx,km_clusters)
            transformed_km_arand_scores[i][k-2]=metrics.adjusted_rand_score(y,km_clusters)
            transformed_km_ami_scores[i][k-2]=metrics.adjusted_mutual_info_score(y,km_clusters)

    meansil = np.mean(sil_scores,axis=0)
    stdsil = np.std(sil_scores,axis=0)

    km_meansil = np.mean(km_sil_scores,axis=0)
    km_stdsil = np.std(km_sil_scores,axis=0)

    transformed_meansil = np.mean(transformed_sil_scores,axis=0)
    transformed_stdsil = np.std(transformed_sil_scores,axis=0)

    transformed_km_meansil = np.mean(transformed_km_sil_scores,axis=0)
    transformed_km_stdsil = np.std(transformed_km_sil_scores,axis=0)

    plt.clf()
    fig, ax = plt.subplots(ncols=1, nrows=1)
    plt.plot(rang,meansil, color="darkorange", label = "EM")
    plt.plot(rang,transformed_meansil, color="r", label = "ICA EM")
    plt.plot(rang,km_meansil, color="b", label="KMeans")
    plt.plot(rang,transformed_km_meansil, color="g", label="ICA KMeans")
    # plt.fill_between(rang, meansil - stdsil,
    #                  meansil + stdsil, alpha=0.1,
    #                  color="r")
    plt.title("ICA Silhouette Score "+datnam)
    plt.ylabel("Silhouette Score")
    plt.xlabel("k")
    ax.grid(b=True)
    plt.tight_layout()
    plt.legend()
    fig.savefig("figs/ica/sil_"+datnam+".png")


    plt.clf()
    mean_arand= np.mean(arand_scores,axis=0)
    mean_ami= np.mean(ami_scores,axis=0)
    km_mean_arand= np.mean(km_arand_scores,axis=0)
    km_mean_ami= np.mean(km_ami_scores,axis=0)
    transformed_mean_arand= np.mean(transformed_arand_scores,axis=0)
    transformed_mean_ami= np.mean(transformed_ami_scores,axis=0)
    transformed_km_mean_arand= np.mean(transformed_km_arand_scores,axis=0)
    transformed_km_mean_ami= np.mean(transformed_km_ami_scores,axis=0)
    plt.plot(rang,mean_ami,color="darkorange", label="EM Adjusted Mutual Information")
    plt.plot(rang,transformed_mean_ami,color="r", label="ICA EM Adjusted Mutual Information" )
    plt.plot(rang,km_mean_ami,color="b", label = "KMeans Adjusted Mutual Information")
    plt.plot(rang,transformed_km_mean_ami,color="g", label="ICA KMeans Adjusted Mutual Information")
    plt.legend()
    plt.ylabel("Adjusted Mutual Information Score")
    plt.xlabel("k")
    plt.title("ICA Adjusted Mutual Information Scores "+datnam)
    plt.savefig("figs/ica/ami"+datnam+".png")


    plt.clf()
    mean_em_time= np.mean(em_time,axis=0)
    mean_km_time= np.mean(km_time,axis=0)
    transformed_mean_em_time= np.mean(transformed_em_time,axis=0)
    transformed_mean_km_time= np.mean(transformed_km_time,axis=0)
    plt.plot(rang,mean_em_time,color="darkorange", label="EM")
    plt.plot(rang,transformed_mean_em_time,color="r", label="ICA EM" )
    plt.plot(rang,mean_km_time,color="b", label = "KMeans")
    plt.plot(rang,transformed_mean_km_time,color="g", label="ICA KMeans")
    plt.legend()
    plt.ylabel("Runtime(s)")
    plt.xlabel("k")
    plt.title("Wall Clock Times "+ datnam)
    plt.savefig("figs/ica/times"+datnam+".png")
    return

if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()