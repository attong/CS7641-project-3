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

def vary_k(x, maxk,title,filnam):
    scores=[]

    rang = np.arange(1,maxk,1)
    for k in rang:
        gm=GaussianMixture(n_components=k)
        gm.fit(x)
        scores.append(gm.score(x))
    plt.clf()
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(rang,scores)
    plt.title(title)
    plt.ylabel("Per Sample Average Log-likelihood")
    plt.xlabel("k")
    ax.grid(b=True)
    plt.tight_layout()
    fig.savefig(filnam)
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
    cv_types = ['spherical', 'tied', 'diag', 'full']
    # vary_k(xtrain1, 50, "Per Sample Average Log-likelihood vs K dat1","figs/em/score_dat1.png" )
    # vary_k(xtrain2, 20, "Per Sample Average Log-likelihood vs K dat2","figs/em/score_dat2.png" )
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