from sklearn.cluster import KMeans

import numpy as np

import csvInOut

data = csvInOut.getData('taxonomySubtreeFreqs3levels2.csv')

km = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=1, verbose=True)

km.fit(data[1:])

labels = data[0]
clusterNum = 1
index = 0

for center in km.cluster_centers_:
    index = 0
    print('Cluster number ' + str(clusterNum))
    clusterNum = clusterNum + 1
    nextString = ''
    for val in center:
        if val > 0.2:
            nextString = nextString + ', ' + labels[index]
        index = index + 1
    print('Significant keywords: ')
    print(nextString)

