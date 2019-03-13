# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:24:45 2019

@author: antho
"""

from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

data = pd.read_csv('centre_com.csv')
cols = data.columns
print(cols)
X= data.iloc[:,[3,4]]

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow curve to find the right number of clusters
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
#5 clusters 

kmeanModel = KMeans(n_clusters=5).fit(X)
kmeanModel.fit(X)

labels = kmeanModel.labels_

data = pd.DataFrame(data,columns=['CustomerID', 'Genre', 'Age', 'Annual Income (k$)','Spending Score (1-100)','Cluster_label'])
data['Cluster_label'] = labels

sns.scatterplot(data.iloc[:,3],data.iloc[:,4],hue=data.iloc[:,5])






















