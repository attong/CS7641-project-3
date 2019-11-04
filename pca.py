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
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import time

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
    plt.plot(rang,transformed_meansil, color="r", label = "PCA EM")
    plt.plot(rang,km_meansil, color="b", label="KMeans")
    plt.plot(rang,transformed_km_meansil, color="g", label="PCA KMeans")
    # plt.fill_between(rang, meansil - stdsil,
    #                  meansil + stdsil, alpha=0.1,
    #                  color="r")
    plt.title("PCA Silhouette Score "+datnam)
    plt.ylabel("Silhouette Score")
    plt.xlabel("k")
    ax.grid(b=True)
    plt.tight_layout()
    plt.legend()
    fig.savefig("figs/pca/sil_"+datnam+".png")


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
    plt.plot(rang,transformed_mean_ami,color="r", label="PCA EM Adjusted Mutual Information" )
    plt.plot(rang,km_mean_ami,color="b", label = "KMeans Adjusted Mutual Information")
    plt.plot(rang,transformed_km_mean_ami,color="g", label="PCA KMeans Adjusted Mutual Information")
    plt.legend()
    plt.ylabel("Adjusted Mutual Information Score")
    plt.xlabel("k")
    plt.title("PCA Adjusted Mutual Information Scores "+datnam)
    plt.savefig("figs/pca/ami"+datnam+".png")


    plt.clf()
    mean_em_time= np.mean(em_time,axis=0)
    mean_km_time= np.mean(km_time,axis=0)
    transformed_mean_em_time= np.mean(transformed_em_time,axis=0)
    transformed_mean_km_time= np.mean(transformed_km_time,axis=0)
    plt.plot(rang,mean_em_time,color="darkorange", label="EM")
    plt.plot(rang,transformed_mean_em_time,color="r", label="PCA EM" )
    plt.plot(rang,mean_km_time,color="b", label = "KMeans")
    plt.plot(rang,transformed_mean_km_time,color="g", label="PCA KMeans")
    plt.legend()
    plt.ylabel("Runtime(s)")
    plt.xlabel("k")
    plt.title("Wall Clock Times "+ datnam)
    plt.savefig("figs/pca/times"+datnam+".png")
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
    # eigenratioplot(xtrain2, "Cum Sum Explained Variance Ratio per Component dat2", "figs/pca/explainedvar_dat2.png")
    # eigenratioplot(xtrain1, "Cum Sum Explained Variance Ratio per Component dat1", "figs/pca/explainedvar_dat1.png")
    # rec_err_plot(xtrain2,18, "Reconstruction Error dat 2", "figs/pca/recon_err_dat2.png")
    # rec_err_plot(xtrain1,54, "Reconstruction Error dat 1", "figs/pca/recon_err_dat1.png")
    pca = PCA(n_components=7)
    data=pca.fit_transform(xtrain2)
    vary_k(xtrain2,data, 20, ytrain2, "dat2")
    pca = PCA(n_components=44)
    data=pca.fit_transform(xtrain1)
    vary_k(xtrain1,data, 50, ytrain1, "dat1")



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