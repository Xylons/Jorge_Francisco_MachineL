# -*- coding: utf-8 -*-

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.cluster import KMeans
from sklearn import metrics 
from sklearn.decomposition import PCA
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import numpy
import collections

iterations = 10
max_iter = 300 
tol = 1e-04 
random_state = 0
init = "random"

#--------------------FUNCTIONS---------------------#

#PCA analysis method
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

#Silhouette and Distorsion method
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

#Kmeans clustering method
def Kmeans(X_pca, k, isPCA=True, normalize=True, bidimensional=False):
    if(not isPCA):
        X_pca = X_pca.to_numpy()
    if(normalize):
        scaler = preprocessing.StandardScaler()
        X_pca = scaler.fit_transform(X_pca)
        
    km = KMeans(k, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(X_pca)

    map = collections.Counter(labels)

    from sklearn import metrics
    print ("Silhouette Score:\n"+str(metrics.silhouette_score(X_pca, labels)))

    print("\nCentroids with number of ocurrences:")
    for x in range(0,numpy.size(km.cluster_centers_,0)-1):
        # print(str(km.cluster_centers_[x])+'\t\tlabel: '+str(x)+' number of ocurrences: '+str(map[x]))
        print('{:<40s} {:<30s}'.format(str(km.cluster_centers_[x]), 'label: '+str(x)+' number of ocurrences: '+str(map[x])))

    if (bidimensional):
        plt.xlabel('Pitch')
        plt.ylabel('Roll')
        x = X_pca[:,0]
        y = X_pca[:,1]
        plt.scatter(x,y, c = labels)
        # plotting centroids
        plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='red',s=50)
        plt.show()
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
        x = X_pca[:,0]
        y = X_pca[:,1]
        z = X_pca[:,2]
        # plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], c='red',s=50)# 
        ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], c='red',s=50)
        ax.scatter(x,y,z, c = labels)
        plt.show()

#Hierarchical clusteing method
def HRC(df, original):
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
    clusters = cluster.hierarchy.linkage(matsim, method = 'ward')
    # http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    # color_threshold a 16 para coger 10 clusters en OrientacionMEAN .4
    # color_threshold a 18 para coger 7 clusters en OrientacionMEAN .4
    # color_threshold a 10 para coger 7 clusters en OrientacionMEAN .4 X_PCA
    cluster.hierarchy.dendrogram(clusters, color_threshold=16)
    plt.show()
    cut = 16 # !!!! ad-hoc
    labels = cluster.hierarchy.fcluster(clusters, cut , criterion = 'distance')
    print ('Number of clusters %d' % (len(set(labels))))
    print (labels)

    colors = numpy.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = numpy.hstack([colors] * 20)

    fig, ax = plt.subplots()
    plt.xlim(-1, 1.5)
    plt.ylim(-1.5, 1)

    for i in range(len(X_pca)):
        plt.text(X_pca[i][0], X_pca[i][1], 'x', color=colors[labels[i]])  
        
    ax.grid(True)
    fig.tight_layout()
    plt.show()

    #3D Plot
    fig = plt.figure()
    ax = Axes3D(fig)
    x = X_pca[:,0]
    y = X_pca[:,1]
    z = X_pca[:,2]
    ax.scatter(x,y,z, c = labels)
    plt.show()

    #Numpy array to dataframe
    data = pd.DataFrame({'Column1': X_pca[:, 0], 'Column2': X_pca[:, 1], 'Column3': X_pca[:, 2]})
    #Mean data by group
    print("\nNormalized data means by group")
    print(data.groupby(labels).mean())

    #Plot with original data and labeled
    fig = plt.figure()
    ax = Axes3D(fig)
    x = original.iloc[:,0]
    y = original.iloc[:,1]
    z = original.iloc[:,2]
    ax.set_xlabel("azimut")
    ax.set_ylabel("pitch")
    ax.set_zlabel("roll")
    ax.scatter(x,y,z, c = labels)
    plt.show()

    #Dataframe to numpy to pass labels
    original = original.to_numpy()
    for i in range(len(original)):
        plt.text(original[i][0], original[i][1], 'x', color=colors[labels[i]]) 
    #Back to dataframe to print group means
    original = pd.DataFrame({'Azimut': original[:, 0], 'Pitch': original[:, 1], 'Roll': original[:, 2]})
    print("\nOriginal data means by group")
    print(original.groupby(labels).mean())

# Data filtering method
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
    # df2daysf.to_csv("T2_clean.csv",mode = 'w', index=False)
    return df2daysf

#Simple plotting method with labels
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


###########################################################


def main():
    #0 . Load the data 
    # read the csv
    df = pd.read_csv("T2_clean.csv")
    # df = pd.read_csv("T2_2days.csv")

    #If we wnat to drop first column [Azimut] for different analysis
    df = df.drop(df.columns[0], axis=1)
    # list the columns
    list(df)

    #Choose function

    #Clean Data
    # df = cleanData(df)

    #Plot Data
    # plotData3D(df)

    #PCA
    # X_pca = transformPCA(df, 3)

    #Silhouette and Distorsion
    # Krange(df)

    #KMeans - best k for OrientationMEAN 7 or 10
    #Kmeans(
    #       Dataframe, 
    #       Number of clusters, 
    #       If the dataframe comes from PCA,
    #       If normalizing is needed,
    #       If the df is bidimesional)
    Kmeans(df, 11, False, False, True)

    #Hierarchical
    #HRC(Dataframe to use PCA or not, Original df to represent labels on)
    # HRC(df, df)

if __name__ == "__main__":
     main()