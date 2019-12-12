import numpy as np
import pandas as pd

def Kmeans(X_pca, k, isPCA=True, normalize=True, bidimensional=False):
    """ 
        Kmeans clustering method
        X_pca = dataset with PCA
        k = number of clusters
        isPCA = if the dataset comes from PCA
        normalize = if normalizing is needed
        bidimensional = if the dataset has only 2 variables
    """

    from mpl_toolkits.mplot3d import Axes3D  
    from sklearn.cluster import KMeans
    from sklearn import preprocessing 
    import matplotlib.pyplot as plt
    import numpy as np
    import collections

    iterations = 10
    max_iter = 300 
    tol = 1e-04 
    random_state = 0
    init = "random"

    if(not isPCA):
        X_pca = X_pca.to_numpy()
    if(normalize):
        scaler = preprocessing.StandardScaler()
        X_pca = scaler.fit_transform(X_pca)
        
    km = KMeans(k, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
    labels = km.fit_predict(X_pca)

    map = collections.Counter(labels)
    pd.DataFrame(km.cluster_centers_).to_csv("centroids.csv")

    from sklearn import metrics
    print ("Silhouette Score:\n"+str(metrics.silhouette_score(X_pca, labels)))

    print("\nCentroids with number of ocurrences:")
    for x in range(0,np.size(km.cluster_centers_,0)):
        # print(str(km.cluster_centers_[x])+'\t\tlabel: '+str(x)+' number of ocurrences: '+str(map[x]))
        print('{:<40s} {:<30s}'.format(str(km.cluster_centers_[x]), 'label: '+str(x)+' number of ocurrences: '+str(map[x])))

    if (bidimensional):
        plt.xlabel('Roll')
        plt.ylabel('Pitch')
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

df = pd.read_csv('task3_dataset_noattacks.csv') 
df = df.drop(df.columns[[0, 1, 2, 3, df.columns.size-1]], axis=1)

Kmeans(df, 10, False, False, True)