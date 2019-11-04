#principal component analysis
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
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from scipy.linalg import pinv
from sklearn.mixture import GaussianMixture
import time


def main():
    #load normalized data
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    #dat1: 36, dat2: 3
    rec_err_plot(xtrain2,18, "Random Projection Reconstruction Error dat2", "figs/rca/recondat2.png")
    rec_err_plot(xtrain1,54, "Random Projection Reconstruction Error dat1", "figs/rca/recondat1.png")
    sil_plots(xtrain2, 18,4,"dat2")
    sil_plots(xtrain1, 54,44,"dat1")
    """0.03288655685198534
    0.017619964663429535
    0.007523731695415414
    0.005271076420350498"""
    transformer=GaussianRandomProjection(n_components=14)
    data = transformer.fit_transform(xtrain2)
    vary_k(xtrain2,data, 20, ytrain2, "dat2")
    transformer=GaussianRandomProjection(n_components=44)
    data = transformer.fit_transform(xtrain1)
    vary_k(xtrain1,data, 50, ytrain1, "dat1")

def inverse_transform(x,transformer):
    W = transformer.components_
    p = pinv(W)
    reconstructed = ((p@W)@(x.T)).T
    return reconstructed

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
            transformed_km_sil_scores[i][k-2]=metrics.silhouette_score(x,km_clusters)
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
    plt.plot(rang,transformed_meansil, color="r", label = "RCA EM")
    plt.plot(rang,km_meansil, color="b", label="KMeans")
    plt.plot(rang,transformed_km_meansil, color="g", label="RCA KMeans")
    # plt.fill_between(rang, meansil - stdsil,
    #                  meansil + stdsil, alpha=0.1,
    #                  color="r")
    plt.title("RCA Silhouette Score "+datnam)
    plt.ylabel("Silhouette Score")
    plt.xlabel("k")
    ax.grid(b=True)
    plt.tight_layout()
    plt.legend()
    fig.savefig("figs/rca/sil_"+datnam+".png")


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
    plt.plot(rang,transformed_mean_ami,color="r", label="RCA EM Adjusted Mutual Information" )
    plt.plot(rang,km_mean_ami,color="b", label = "KMeans Adjusted Mutual Information")
    plt.plot(rang,transformed_km_mean_ami,color="g", label="RCA KMeans Adjusted Mutual Information")
    plt.legend()
    plt.ylabel("Adjusted Mutual Information Score")
    plt.xlabel("k")
    plt.title("RCA Adjusted Mutual Information Scores "+datnam)
    plt.savefig("figs/rca/ami"+datnam+".png")


    plt.clf()
    mean_em_time= np.mean(em_time,axis=0)
    mean_km_time= np.mean(km_time,axis=0)
    transformed_mean_em_time= np.mean(transformed_em_time,axis=0)
    transformed_mean_km_time= np.mean(transformed_km_time,axis=0)
    plt.plot(rang,mean_em_time,color="darkorange", label="EM")
    plt.plot(rang,transformed_mean_em_time,color="r", label="RCA EM" )
    plt.plot(rang,mean_km_time,color="b", label = "KMeans")
    plt.plot(rang,transformed_mean_km_time,color="g", label="RCA KMeans")
    plt.legend()
    plt.ylabel("Runtime(s)")
    plt.xlabel("k")
    plt.title("Wall Clock Times "+ datnam)
    plt.savefig("figs/rca/times"+datnam+".png")
    return


def sil_plots(x,comps, numclusters, datnam,reps=10):
    plt.clf()
    components = np.arange(1,comps+1,1)
    km_sil_scores = np.ones((reps,comps))
    standard = np.ones((reps,1))
    for i in range(reps):
        km=KMeans(n_clusters= numclusters)
        clusters = km.fit_predict(x)
        standard[i][0]=metrics.silhouette_score(x,clusters,metric="euclidean")
        for j in components:    
            transformer=GaussianRandomProjection(n_components=j)
            data = transformer.fit_transform(x)
            km = KMeans(n_clusters = numclusters)
            clusters = km.fit_predict(data)
            sil = metrics.silhouette_score(data,clusters,metric="euclidean")
            km_sil_scores[i][j-1]=sil
            # temp.append(sil)
        # plt.plot(components,temp)
    m = np.mean(km_sil_scores,axis=0)
    stderr = np.std(km_sil_scores,axis=0)
    print(np.mean(stderr))
    s_m = np.mean(standard)
    s_std = np.std(standard)
    print(s_std)
    plt.hlines(s_m,0,comps, linestyle="dashed", label="Non-reduced Kmeans Score")
    plt.fill_between(components, s_m - s_std,
                     s_m + s_std, alpha=0.1,
                     color="black")
    plt.plot(components,m,'o-', color="r",label="Mean silhouette score",linewidth=4.0)
    plt.fill_between(components, m - stderr,
                     m + stderr, alpha=0.1,
                     color="r")
    plt.ylabel("Silhouette Score")
    plt.xlabel("Number of Components")
    plt.legend()
    # plt.tight_layout()
    plt.title("RDA Silhouette Scores "+datnam)
    plt.savefig("figs/rca/sil"+datnam)

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