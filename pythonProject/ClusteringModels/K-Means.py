import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# load the training dataset
data = pd.read_csv('seeds.csv')
# Display a random sample of 10 observations (just the features)
features = data[data.columns[0:6]]
features.sample(10)

# Normalize the numeric features so they're on the same scale
scaled_features = MinMaxScaler().fit_transform(features[data.columns[0:6]])

# Get two principal components
pca = PCA(n_components=2).fit(scaled_features)
features_2d = pca.transform(scaled_features)
print(features_2d[0:10])

# Create a model based on 3 centroids
model = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
# Fit to the data and predict the cluster assignments for each data point
km_clusters = model.fit_predict(features.values)
# View the cluster assignments
print(km_clusters)


def plot_clusters(samples, clusters):
    col_dic = {0: 'blue', 1: 'green', 2: 'orange'}
    mrk_dic = {0: '*', 1: 'x', 2: '+'}
    colors = [col_dic[x] for x in clusters]
    markers = [mrk_dic[x] for x in clusters]
    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1], color=colors[sample], marker=markers[sample], s=100)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Assignments')
    plt.show()


plot_clusters(features_2d, km_clusters)

seed_species = data[data.columns[7]]
plot_clusters(features_2d, seed_species.values)
