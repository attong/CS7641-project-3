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
from kmeans import elbowplot
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import time
from sklearn.decomposition import FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

def reduction_nn():
    out="csv output/"
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data(test_size=0.05)
    nn2 = MLPClassifier(activation= 'relu', alpha= 0.001, hidden_layer_sizes= (140,), learning_rate_init= 0.0033333366666666664)
    gs = GridSearchCV(nn2,{},return_train_score=True,verbose=10,cv=5)
    gs.fit(xtrain2,ytrain2)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'dat2.csv')

    nn2 = MLPClassifier(activation= 'relu', alpha= 0.001, hidden_layer_sizes= (140,), learning_rate_init= 0.0033333366666666664)
    pca=PCA()
    pipe = Pipeline(steps=[('pca', pca), ('neuralnet', nn2)])
    grid = {'pca__n_components': np.arange(1,19,1)}
    gs = GridSearchCV(pipe,grid,return_train_score=True,verbose=2,cv=5)
    gs.fit(xtrain2,ytrain2)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'pcadat2.csv')

    nn2 = MLPClassifier(activation= 'relu', alpha= 0.001, hidden_layer_sizes= (140,), learning_rate_init= 0.0033333366666666664)
    ica = FastICA()
    pipe = Pipeline(steps=[('ica', ica), ('neuralnet', nn2)])
    grid = {'ica__n_components': np.arange(1,19,1)}
    gs = GridSearchCV(pipe,grid,return_train_score=True,verbose=10,cv=5)
    gs.fit(xtrain2,ytrain2)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'icadat2.csv')

    nn2 = MLPClassifier(activation= 'relu', alpha= 0.001, hidden_layer_sizes= (140,), learning_rate_init= 0.0033333366666666664)
    rca = GaussianRandomProjection()
    pipe = Pipeline(steps=[('rca', rca), ('neuralnet', nn2)])
    grid = {'rca__n_components': np.arange(1,19,1)}
    gs = GridSearchCV(pipe,grid,return_train_score=True,verbose=10,cv=5)
    gs.fit(xtrain2,ytrain2)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'rcadat2.csv')

    nn2 = MLPClassifier(activation= 'relu', alpha= 0.001, hidden_layer_sizes= (140,), learning_rate_init= 0.0033333366666666664)

    lda = LinearDiscriminantAnalysis()
    pipe = Pipeline(steps=[('lda', lda), ('neuralnet', nn2)])
    grid = {'lda__n_components': np.arange(1,4,1)}
    gs = GridSearchCV(pipe,grid,return_train_score=True,verbose=10,cv=5)
    gs.fit(xtrain2,ytrain2)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'ldadat2.csv')

#taken from here to help with pipelining
#https://github.com/JonathanTay/CS-7641-assignment-3/blob/d188918a5a9b5e3b12d662725ee44d53bb7e48e2/helpers.py#L34
class myGMM(GaussianMixture):
    def transform(self,X):
        return self.predict_proba(X)

class myGMMstack(GaussianMixture):
    def transform(self,X):
        return np.hstack((X,self.predict_proba(X)))

def kmeanstack():
    frame = np.ones((5,20))
    times = np.ones((5,20))
    for j in range(5):
        xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data(test_size=0.2)
        for i in range(1,20,1):
            km=KMeans(n_clusters=i)
            xnew= np.hstack((xtrain2,km.fit_transform(xtrain2)))
            xtestnew = np.hstack((xtest2, km.transform(xtest2)))
            nn2 = MLPClassifier(activation= 'relu', alpha= 0.001, hidden_layer_sizes= (140,), learning_rate_init= 0.0033333366666666664)
            start = time.time()
            nn2.fit(xnew,ytrain2)
            fittime=time.time()-start
            times[j][i-1]= fittime
            frame[j][i-1]= nn2.score(xtestnew,ytest2)
    np.savetxt("nnkmstack.csv", frame, delimiter=",")
    np.savetxt("nnkmstacktimes.csv", times, delimiter=",")


def featurize_cluster(x):
    for i in range(1,2,1):
        km=KMeans(n_clusters=i)
        print(np.hstack((x,km.fit_transform(x))))

def cluster_nn():
    out="csv output/"
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data(test_size=0.05)
    nn2 = MLPClassifier(activation= 'relu', alpha= 0.001, hidden_layer_sizes= (140,), learning_rate_init= 0.0033333366666666664)
    km= KMeans()
    pipe=Pipeline(steps=[('km', km), ('neuralnet', nn2)])
    grid = {'km__n_clusters': np.arange(1,20,1)}
    gs = GridSearchCV(pipe,grid,return_train_score=True,verbose=10,cv=5)
    gs.fit(xtrain2,ytrain2)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'kmnndat2.csv')

    nn2 = MLPClassifier(activation= 'relu', alpha= 0.001, hidden_layer_sizes= (140,), learning_rate_init= 0.0033333366666666664)
    em =myGMM()
    pipe=Pipeline(steps=[('em', em), ('neuralnet', nn2)])
    grid = {'em__n_components': np.arange(1,20,1)}
    gs = GridSearchCV(pipe,grid,return_train_score=True,verbose=10,cv=5)
    gs.fit(xtrain2,ytrain2)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'emnndat2.csv')

    nn2 = MLPClassifier(activation= 'relu', alpha= 0.001, hidden_layer_sizes= (140,), learning_rate_init= 0.0033333366666666664)
    em =myGMMstack()
    pipe=Pipeline(steps=[('em', em), ('neuralnet', nn2)])
    grid = {'em__n_components': np.arange(1,20,1)}
    gs = GridSearchCV(pipe,grid,return_train_score=True,verbose=10,cv=5)
    gs.fit(xtrain2,ytrain2)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'emstacknndat2.csv')

    kmeanstack()



def main():
    reduction_nn()
    cluster_nn()
    # xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data(test_size=0.05)
    # featurize_cluster(xtrain2)

if __name__ == '__main__':
    np.random.seed()
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)
    main()