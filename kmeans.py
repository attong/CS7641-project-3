from app import load_data
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.manifold import TSNE

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

def elbowplot(x, k, metric, title, fignam, elbow=True):
    plt.clf()
    km = KMeans()
    visualizer= KElbowVisualizer(km, k=k, metric=metric, timings= False, title=title, locate_elbow=elbow)
    visualizer.fit(x)
    visualizer.show(outpath=fignam)
    return



def main():
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
    km= KMeans(4)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
    visualizer.fit(xtrain2)
    visualizer.show()
    plt.clf()
    fig, ax=plt.subplots()
    tsne= TSNE()
    Y=tsne.fit_transform(xtrain2)  
    ax.scatter(Y[:, 0], Y[:, 1], c='b')
    plt.show()
    # elbowplot(xtrain2,20,"distortion","K Means Clustering Distortion vs Number of Clusters dat2","figs/kmeans/kmeans_elbow_dat2.png")
    # elbowplot(xtrain1,100,"distortion","K Means Clustering Distortion vs Number of Clusters dat1","figs/kmeans/kmeans_elbow_dat1.png")
    # elbowplot(xtrain2,20,"silhouette","K Means Clustering Silhouette Score vs Number of Clusters dat2","figs/kmeans/kmeans_silhouette_dat2.png", elbow=False)
    # elbowplot(xtrain1,100,"silhouette","K Means Clustering Silhouette Score vs Number of Clusters dat1","figs/kmeans/kmeans_silhouette_dat1.png",elbow=False)
    # elbowplot(xtrain2,20,"calinski_harabasz","K Means Clustering Calinski Harabasz Score vs Number of Clusters dat2","figs/kmeans/kmeans_calinski_dat2.png", elbow=False)
    # elbowplot(xtrain1,100,"calinski_harabasz","K Means Clustering Calinski Harabasz Score vs Number of Clusters dat1","figs/kmeans/kmeans_calinski_dat1.png",elbow=False)



if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()