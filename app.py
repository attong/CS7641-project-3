#@author: anthony tong

import numpy as np
import pandas as pd
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split

def load_dat2():
    filnames=['xaa.dat','xab.dat','xac.dat','xad.dat','xae.dat','xaf.dat','xag.dat','xah.dat','xai.dat']
    names = ['COMPACTNESS','CIRCULARITY','DISTANCE CIRCULARITY','RADIUS RATIO','PR.AXIS ASPECT RATIO','MAX.LENGTH ASPECT RATIO','SCATTER RATIO','ELONGATEDNESS area/(shrink width)**2',
    'PR.AXIS RECTANGULARITY','MAX.LENGTH RECTANGULARITY', 'SCALED VARIANCE ALONG MAJOR AXIS','SCALED VARIANCE ALONG MINOR AXIS',
    'SCALED RADIUS OF GYRATION','SKEWNESS ABOUT MAJOR AXIS','SKEWNESS ABOUT MINOR AXIS',
    'KURTOSIS ABOUT MINOR AXIS','KURTOSIS ABOUT MAJOR AXIS', 'HOLLOWS RATIO','type']
    li=[]
    for i in filnames:
        df = pd.read_csv('data/vehicle_silouettes/{}'.format(i),names=names,sep="\s+")
        li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    y = df.pop('type')
    y = y.astype('category')
    for col in df.columns:
        df.loc[:,col] = df[col].astype(float)
    return df, y

def load_dat1():
    df= pd.read_csv("data/online_shoppers_intention.csv")
    #convert numerical columns to floats
    numeric_cols = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration', 'BounceRates','ExitRates','PageValues','SpecialDay']
    #convert categorical columns to numbers
    #split 50% into test and train set
    rev = df.pop('Revenue').values
    categorical_cols = [x for x in df.columns if x not in numeric_cols]
    for col in categorical_cols:
        df.loc[:,col] = df[col].astype('category')
    for col in numeric_cols:
        df.loc[:,col] = df[col].astype(float)
    df= pd.get_dummies(df)
    return df,rev

def normalize(train,test):
    scaler = StandardScaler()
    scaler.fit(train)
    train=scaler.transform(train)
    test=scaler.transform(test)
    return train,test
    
def load_data(test_size=0.2):
    df, rev = load_dat1()
    df2, y2 = load_dat2()
    xtrain1, xtest1, ytrain1, ytest1= train_test_split(df,rev,test_size=test_size)
    xtrain2, xtest2, ytrain2, ytest2= train_test_split(df2,y2,test_size=test_size)
    xtrain1, xtest1 = normalize(xtrain1,xtest1)
    xtrain2, xtest2 = normalize(xtrain2,xtest2)
    return xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2

def main():
    xtrain1, xtest1, ytrain1, ytest1, xtrain2, xtest2, ytrain2, ytest2 = load_data()
