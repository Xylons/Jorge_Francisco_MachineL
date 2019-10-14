# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

import pandas as pd
from mpl_toolkits.mplot3d import Axes3D   

#0 . Load the data 
# read the csv
df = pd.read_csv("T2.csv")
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
#filter data by date
df04 = df[(df['TimeStemp'] > '2016-05-04 00:00:00') & (df['TimeStemp'] <= '2016-05-04 23:59:59')]
df11 = df[(df['TimeStemp'] > '2016-05-11 00:00:00') & (df['TimeStemp'] <= '2016-05-11 23:59:59')]
#group both days of data
frames = [df04, df11]
#concatenate both days of data
df2days = pd.concat(frames)

print (df2days.shape)


#1.2. Filter Features

# we want only the *MEAN features from all the sensors.
#https://stackoverflow.com/questions/30808430/how-to-select-columns-from-dataframe-by-regex
# df2daysf = df2days.filter(regex=("*MEAN"))

# Gyroscope
# df2daysf = df2days[[c for c in df if c.startswith('GyroscopeStat') and c.endswith('MEAN')]]
# Orientation
# df2daysf = df2days[[c for c in df if c.startswith('OrientationProbe')]]
# Pressure
df2daysf = df2days[[c for c in df if c.startswith('Pressure')]]
# Accelerometer
# df2daysf = df2days[[c for c in df if c.startswith('AccelerometerStat') and c.endswith('MEAN')]]
# LinearAcceleration
# df2daysf = df2days[[c for c in df if c.startswith('LinearAcceleration') and c.endswith('MEAN')]]
# All means
# df2daysf = df2daysf[[c for c in df if c.endswith('MEAN')]]
list(df2daysf)

# RotationVector_cosThetaOver2_MEAN is a feature with all values as NaN
# exclude = ["RotationVector_cosThetaOver2_MEAN"]
# df2daysf = df2daysf.loc[:, df2daysf.columns.difference(exclude)]

# 1.3 remove missing values
df2daysf.isnull().values.any()
# filter/remove rows with missing values (na) (Be careful!!!)
df2daysf = df2daysf.dropna()
df2daysf.isnull().values.any()

# 1.4 trim data frequency
# deleting even rows to reduce frequency data by half
df2daysf=df2daysf[::2]
# deleting even rows to reduce frequency data by four
# df2daysf=df2daysf[::4]

print (df2daysf.shape)

# 2. Principal Component Analysis
#2.1 Scalation
from sklearn import preprocessing 
scaler = preprocessing.StandardScaler()
datanorm = scaler.fit_transform(df2daysf)

#2.2 Modelling (PCA)
from sklearn.decomposition import PCA
n_components = 3
estimator = PCA (n_components)
X_pca = estimator.fit_transform(datanorm)

# is it representative the 2D projection?
print (estimator.explained_variance_ratio_)


#2.3 Plot 
import matplotlib.pyplot as plt
import numpy

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


# Clustering
from sklearn.cluster import KMeans

iterations = 10
max_iter = 300 
tol = 1e-04 
random_state = 0
k = 50
init = "random"
km = KMeans(k, init, n_init = iterations ,max_iter= max_iter, tol = tol,random_state = random_state)
labels = km.fit_predict(df2daysf)

from sklearn import metrics
print (metrics.silhouette_score(df2daysf, labels))

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
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], km.cluster_centers_[:,2], c='red')#,s=50)
ax.scatter(x,y,z, c = labels)
plt.show()
