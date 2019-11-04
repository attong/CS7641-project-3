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
from sklearn.mixture import GaussianMixture
import time


def main():
    #load normalized data
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    varianceratioplot(xtrain2,ytrain2,"LDA Cum Sum Variance dat2","figs/lda/varianceratiodat2.png")
    varianceratioplot(xtrain1,ytrain1,"LDA Cum Sum Variance dat1","figs/lda/varianceratiodat1.png")
    lda = LinearDiscriminantAnalysis()
    data = lda.fit_transform(xtrain2,ytrain2)
    vary_k(xtrain2,data, 20, ytrain2, "dat2")
    plt.clf()
    data = lda.fit_transform(xtrain1,ytrain1)

    vary_k(xtrain1,data, 50, ytrain1, "dat1", iters=2)
    # model = KMeans(15, random_state=42)
    # visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    # lda = LinearDiscriminantAnalysis()
    # temp=lda.fit_transform(xtrain2,ytrain2)
    # visualizer.fit(temp)
    # visualizer.show(outpath="temp1231.png")
    # temp= np.transpose(data)[0]
    # temp2= np.transpose(data)[1]
    # cdict = {"bus":"yellow","van":"brown","opel":"red",'saab':"blue"}
    # for g in np.unique(ytrain2):
    #     ix = np.where(ytrain2 == g)
    #     plt.scatter(temp[ix], temp2[ix], c = cdict[g], label = g, s = 100, marker="+")
    # plt.savefig("temp.png")
    # plt.clf()
    # data = lda.fit_transform(xtrain1,ytrain1)
    # temp= np.transpose(data)[0]
    # temp2= temp/temp
    # cdict = {True:"green",False:"red"}
    # for g in np.unique(ytrain1):
    #     ix = np.where(ytrain1 == g)
    #     plt.scatter(temp[ix], temp2[ix], c = cdict[g], label = g, s = 100, marker="+")
    # plt.savefig("temp2.png")


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
    plt.plot(rang,transformed_meansil, color="r", label = "LDA EM")
    plt.plot(rang,km_meansil, color="b", label="KMeans")
    plt.plot(rang,transformed_km_meansil, color="g", label="LDA KMeans")
    # plt.fill_between(rang, meansil - stdsil,
    #                  meansil + stdsil, alpha=0.1,
    #                  color="r")
    plt.title("LDA Silhouette Score "+datnam)
    plt.ylabel("Silhouette Score")
    plt.xlabel("k")
    ax.grid(b=True)
    plt.tight_layout()
    plt.legend()
    fig.savefig("figs/lda/sil_"+datnam+".png")


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
    plt.plot(rang,transformed_mean_ami,color="r", label="LDA EM Adjusted Mutual Information" )
    plt.plot(rang,km_mean_ami,color="b", label = "KMeans Adjusted Mutual Information")
    plt.plot(rang,transformed_km_mean_ami,color="g", label="LDA KMeans Adjusted Mutual Information")
    plt.legend()
    plt.ylabel("Adjusted Mutual Information Score")
    plt.xlabel("k")
    plt.title("LDA Adjusted Mutual Information Scores "+datnam)
    plt.savefig("figs/lda/ami"+datnam+".png")


    plt.clf()
    mean_em_time= np.mean(em_time,axis=0)
    mean_km_time= np.mean(km_time,axis=0)
    transformed_mean_em_time= np.mean(transformed_em_time,axis=0)
    transformed_mean_km_time= np.mean(transformed_km_time,axis=0)
    plt.plot(rang,mean_em_time,color="darkorange", label="EM")
    plt.plot(rang,transformed_mean_em_time,color="r", label="LDA EM" )
    plt.plot(rang,mean_km_time,color="b", label = "KMeans")
    plt.plot(rang,transformed_mean_km_time,color="g", label="LDA KMeans")
    plt.legend()
    plt.ylabel("Runtime(s)")
    plt.xlabel("k")
    plt.title("Wall Clock Times "+ datnam)
    plt.savefig("figs/lda/times"+datnam+".png")
    return


if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()