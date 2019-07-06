# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:18:12 2019

@author: Shubham
"""

#libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('Mall_Customers.csv')
data = dataset.drop(['CustomerID'], axis=1)
data['Gender'] = data['Gender'].factorize()[0]

#visualising
sns.pairplot(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]);

g = sns.PairGrid(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=8);

fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(3, 3, 1)
sns.barplot(x= 'Gender', y='Age', data=data)
ax2 = fig.add_subplot(3, 3, 2)
sns.barplot(x= 'Gender', y='Annual Income (k$)', data=data)
ax3 = fig.add_subplot(3, 3, 3)
sns.barplot(x= 'Gender', y='Spending Score (1-100)', data=data)

#pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_new = pca.fit_transform(data)
plt.figure(figsize=(12, 12))
plt.scatter(data_new[:,0], data_new[:,1], c='Red')
plt.show()

from sklearn.manifold import TSNE
tsn = TSNE()
res_tsn = tsn.fit_transform(data)
plt.figure(figsize=(12, 12))
plt.scatter(res_tsn[:,0], res_tsn[:,1], c='Red')
plt.show()

#dendogram
from scipy.cluster.hierarchy import dendrogram, linkage
dendrogram = dendrogram(linkage(data, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


#clustering
from sklearn.cluster import AgglomerativeClustering as AggClus
hc = AggClus(n_clusters = 5)
y_hc = hc.fit_predict(data_new)
plt.scatter(data_new[y_hc == 0, 0], data_new[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(data_new[y_hc == 1, 0], data_new[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(data_new[y_hc == 2, 0], data_new[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(data_new[y_hc == 3, 0], data_new[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(data_new[y_hc == 4, 0], data_new[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#adding result
dataset['Predict'] = pd.DataFrame(y_hc)