from app import load_data
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib
import itertools
matplotlib.use('Agg')
from sklearn import metrics
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans
import time

def vary_k(x, maxk,y,datnam,iters=5):
    scores=[]
    rang = np.arange(2,maxk,1)
    sil_scores = np.ones((iters,maxk-2))
    arand_scores = np.ones((iters,maxk-2))
    ami_scores = np.ones((iters,maxk-2))
    km_sil_scores = np.ones((iters,maxk-2))
    km_arand_scores = np.ones((iters,maxk-2))
    km_ami_scores = np.ones((iters,maxk-2))
    em_time=np.ones((iters,maxk-2))
    km_time = np.ones((iters,maxk-2)) 
    for i in range(iters):
        for k in rang:
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
    meansil = np.mean(sil_scores,axis=0)
    stdsil = np.std(sil_scores,axis=0)

    km_meansil = np.mean(km_sil_scores,axis=0)
    km_stdsil = np.std(km_sil_scores,axis=0)
    plt.clf()
    fig, ax = plt.subplots(ncols=1, nrows=1)
    plt.plot(rang,meansil, color="r", label = "EM")
    plt.plot(rang,km_meansil, color="g", label="KMeans")
    # plt.fill_between(rang, meansil - stdsil,
    #                  meansil + stdsil, alpha=0.1,
    #                  color="r")
    plt.title("Silhouette Score "+datnam)
    plt.ylabel("Silhouette Score")
    plt.xlabel("k")
    ax.grid(b=True)
    plt.tight_layout()
    plt.legend()
    fig.savefig("figs/em/sil_"+datnam+".png")


    plt.clf()
    mean_arand= np.mean(arand_scores,axis=0)
    mean_ami= np.mean(ami_scores,axis=0)
    km_mean_arand= np.mean(km_arand_scores,axis=0)
    km_mean_ami= np.mean(km_ami_scores,axis=0)
    plt.plot(rang,mean_arand,color="darkorange", label="EM Adjusted Random Index")
    plt.plot(rang,mean_ami,color="r", label="EM Adjusted Mutual Information" )
    plt.plot(rang,km_mean_arand,color="b", label = "KMeans Adjusted Random Index")
    plt.plot(rang,km_mean_ami,color="g", label="KMeans Adjusted Mutual Information")
    plt.legend()
    plt.ylabel("Score")
    plt.xlabel("k")
    plt.title("Adjusted Random Index and Mutual Info Scores "+datnam)
    plt.savefig("figs/em/ami_arand"+datnam+".png")


    plt.clf()
    mean_em_time= np.mean(em_time,axis=0)
    mean_km_time= np.mean(km_time,axis=0)
    plt.plot(rang,mean_em_time, color="r", label="EM")
    plt.plot(rang,mean_km_time, color="g", label="KMeans")
    plt.legend()
    plt.ylabel("Runtime(s)")
    plt.xlabel("k")
    plt.title("Wall Clock Times "+ datnam)
    plt.savefig("figs/em/times"+datnam+".png")
    return

#code adapted from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
def bic_model_selection(x,kval, title, filnam,ylabel):
    plt.clf()
    cv_types = ['spherical', 'tied', 'diag', 'full']
    bic=[]
    krange = np.arange(1,kval,1)
    for cv in cv_types:
        for k in krange:
            gm = GaussianMixture(n_components=k,covariance_type = cv)
            gm.fit(x)
            if ylabel == "BIC Score":
                bic.append(gm.bic(x))
            elif ylabel == "AIC Score":
                bic.append(gm.aic(x))
            else:
                bic.append(gm.score(x))
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange'])
    bars=[]
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(krange) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(krange): (i + 1) * len(krange)], width=.2, color=color))
    plt.title(title)
    plt.xlabel('Number of components')
    plt.ylabel(ylabel)
    plt.legend([b[0] for b in bars], cv_types)
    plt.savefig(filnam)
    return



def main():
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    vary_k(xtrain2, 20, ytrain2, "dat2")
    vary_k(xtrain1, 50, ytrain1, "dat1")
    bic_model_selection(xtrain2,20, "BIC per model dat2","figs/em/bic_dat2.png","BIC Score" )
    bic_model_selection(xtrain2,20, "AIC per model dat2","figs/em/aic_dat2.png","AIC Score" )
    bic_model_selection(xtrain2,20, "Average Log likelihood per model dat2","figs/em/score_dat2.png","Average Log Likelihood Score" )
    bic_model_selection(xtrain1,100, "BIC per model dat1","figs/em/bic_dat1.png","BIC Score" )
    bic_model_selection(xtrain1,100, "AIC per model dat1","figs/em/aic_dat1.png","AIC Score" )
    bic_model_selection(xtrain1,100, "Average Log likelihood per model dat1","figs/em/score_dat1.png","Average Log Likelihood Score" )

if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()