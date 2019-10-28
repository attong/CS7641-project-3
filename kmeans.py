from app import load_data
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn import metrics

# metrics.silhouette_score(X, labels, metric='euclidean')
def kplot(x, krange, incre, title, filnam,title_sil,filnam_sil):
    k_values = np.arange(2,krange,incre)
    squared_distances = []
    sil = []
    for k in k_values:
        km = KMeans(n_clusters=k, max_iter=1000, n_init=20)
        labels=km.fit_predict(x)
        squared_distances.append(km.inertia_)
        sil.append(metrics.silhouette_score(x,labels,metric="euclidean"))
    plt.clf()
    plt.plot(k_values,squared_distances)
    plt.ylabel("Sum of Squared Distances to Cluster Centers")
    plt.xlabel("Number of Clusters")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig("figs/"+filnam)
    plt.clf()
    plt.plot(k_values, sil)
    plt.ylabel("Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.grid()
    plt.title(title_sil)
    plt.tight_layout()
    plt.savefig("figs/"+filnam_sil)
    return

def main():
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    kplot(xtrain2, 30, 1, "K means Sum of Squared Distances to Cluster Centers dat2", "kmeans_elbow_dat2.png","K means Silhouette Coefficient dat2", "kmeans_silhouette_dat2.png")
    kplot(xtrain1, 100, 1, "K means Sum of Squared Distances to Cluster Centers dat1", "kmeans_elbow_dat1.png","K means Silhouette Coefficient dat1", "kmeans_silhouette_dat1.png")


if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()