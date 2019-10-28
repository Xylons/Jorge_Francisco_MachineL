# -*- coding: utf-8 -*-

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.cluster import KMeans
from sklearn import metrics 
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import numpy

iterations = 10
max_iter = 300 
tol = 1e-04 
random_state = 0
init = "random"

#--------------------FUNCTIONS---------------------#

def transformPCA(df, n):
    # 2. Principal Component Analysis
    #2.1 Scalation
    
    scaler = preprocessing.StandardScaler()
    datanorm = scaler.fit_transform(df)

    #2.2 Modelling (PCA)
    
    n_components = n
    estimator = PCA (n_components)
    X_pca = estimator.fit_transform(datanorm)

    # is it representative the 2D projection?
    print (estimator.explained_variance_ratio_)

    #2.3 Plot 
    if (n_components >= 2): 
        x = X_pca[:,0]
        y = X_pca[:,1]
        plt.scatter(x,y)
        plt.show()
        

    if (n_components >= 3):
    
        fig = plt.figure()
        ax = Axes3D(fig)
        x = X_pca[:,0]
        y = X_pca[:,1]
        z = X_pca[:,2]
        ax.scatter(x,y,z)
        plt.show()
    
    return X_pca

def Krange(X_pca):

    distortions = []
    silhouettes = []

    for i in range(2, 50):
        km = KMeans(i, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
        labels = km.fit_predict(X_pca)
        distortions.append(km.inertia_)
        silhouettes.append(metrics.silhouette_score(X_pca, labels))
    
    plt.figure(1)
    plt.plot(range(2,50), distortions, marker='o')
    plt.xlabel('K')
    plt.ylabel('Distortion')
    # plt.show()

    plt.figure(2)
    plt.plot(range(2,50), silhouettes , marker='o')
    plt.xlabel('K')
    plt.ylabel('Silhouette')
    plt.show()

def Kmeans(X_pca,k):
    km = KMeans(k, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(X_pca)

    from sklearn import metrics
    print (metrics.silhouette_score(X_pca, labels))

    x = X_pca[:,0]
    y = X_pca[:,1]
    plt.scatter(x,y, c = labels)
    # plotting centroids
    plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='red',s=50)
    plt.show()


    fig = plt.figure()
    ax = Axes3D(fig)
    x = X_pca[:,0]
    y = X_pca[:,1]
    z = X_pca[:,2]
    # plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], c='red',s=50)# Analyze why this fails
    ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], c='red',s=50)
    ax.scatter(x,y,z, c = labels)
    plt.show()

def HRC(df):
    from sklearn import preprocessing 
    min_max_scaler = preprocessing.MinMaxScaler()
    datanorm = min_max_scaler.fit_transform(df)

    from sklearn.decomposition import PCA
    estimator = PCA (n_components = 3)
    X_pca = estimator.fit_transform(datanorm)

    #3. Hierarchical Clustering
    # 3.1. Compute the similarity matrix
    import sklearn.neighbors
    dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
    matsim = dist.pairwise(datanorm)

    # 3.2. Building the Dendrogram	
    from scipy import cluster
    # method linkage: simple, ward, complete
    clusters = cluster.hierarchy.linkage(matsim, method = 'complete')
    # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    # color_threshold a 16 para coger 10 clusters en OrientacionMEAN .4
    # color_threshold a 18 para coger 7 clusters en OrientacionMEAN .4
    cluster.hierarchy.dendrogram(clusters, color_threshold=18)
    plt.show()
    cut = 18 # !!!! ad-hoc
    labels = cluster.hierarchy.fcluster(clusters, cut , criterion = 'distance')
    print ('Number of clusters %d' % (len(set(labels))))
    print (labels)

    colors = numpy.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = numpy.hstack([colors] * 20)

    fig, ax = plt.subplots()
    plt.xlim(-1, 2)
    plt.ylim(-0.5, 1)

    for i in range(len(X_pca)):
        plt.text(X_pca[i][0], X_pca[i][1], 'x', color=colors[labels[i]])  
        
    ax.grid(True)
    fig.tight_layout()
    plt.show()

#--------------------DATA FILTER---------------------#
def cleanData(df):
    #0 . Load the data 
    # read the csv
    # df = pd.read_csv("T2.csv")
    # list the columns
    list(df)
    # print number of rows and columns 
    print (df.shape)

    # 1. Filtering

    # 1.1 Filter rows
    # convert string to datetime .... Be careful!!! Spelling errors!!!
    df['TimeStemp'] = pd.to_datetime(df['TimeStemp'])
    # extract date from datetime
    df['date'] = [d.date() for d in df['TimeStemp']]
    # list the available days
    df['date'].unique()
    #filter data by date. We take first two wednesdays from the dataset
    df04 = df[(df['TimeStemp'] > '2016-05-04 00:00:00') & (df['TimeStemp'] <= '2016-05-04 23:59:59')]
    df11 = df[(df['TimeStemp'] > '2016-05-11 00:00:00') & (df['TimeStemp'] <= '2016-05-11 23:59:59')]
    #group both days of data
    frames = [df04, df11]
    #concatenate both days of data
    df2days = pd.concat(frames)

    print (df2days.shape)

    #1.2. Filter Features

    # Gyroscope
    # df2daysf = df2days[[c for c in df if c.startswith('GyroscopeStat') and c.endswith('MEAN')]]
    # Orientation
    df2daysf = df2days[[c for c in df if c.startswith('OrientationProbe') and c.endswith('MEAN')]]
    # Pressure
    # df2daysf = df2days[[c for c in df if c.startswith('Pressure')]]
    # Accelerometer
    # df2daysf = df2days[[c for c in df if c.startswith('AccelerometerStat') and c.endswith('MEAN')]]
    # LinearAcceleration
    # df2daysf = df2days[[c for c in df if c.startswith('LinearAcceleration') and c.endswith('MEAN')]]
    # All means
    # df2daysf = df2daysf[[c for c in df if c.endswith('MEAN')]]
    list(df2daysf)

    # 1.3 remove missing values
    df2daysf.isnull().values.any()
    # filter/remove rows with missing values (na) (Be careful!!!)
    df2daysf = df2daysf.dropna()
    df2daysf.isnull().values.any()

    # 1.4 trim data frequency
    # deleting even rows to reduce frequency data by half
    # df2daysf=df2daysf[::2]
    # deleting even rows to reduce frequency data by four
    df2daysf=df2daysf[::4]
    #we could also trim with the mean
    print(df2daysf.shape)
    df2daysf.to_csv("T2_clean.csv",mode = 'w', index=False)
    return df2daysf

def plotData3D(df):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    z = df.iloc[:,2]
    ax.scatter(x,y,z)
    ax.set_xlabel("azimut")
    ax.set_ylabel("pitch")
    ax.set_zlabel("roll")
    plt.show()


#0 . Load the data 
# read the csv
df = pd.read_csv("T2_clean.csv")
# list the columns
list(df)

#Choose function

#Clean Data
# df = cleanData(df)

#Plot Data
plotData3D(df)

#PCA
# X_pca = transformPCA(df, 3)

#Kmeans
    # Krange(X_pca)
    # best k for OrientationMEAN 7 or 10
# Kmeans(X_pca, 7)

#Jerarquico
# HRC(df)

