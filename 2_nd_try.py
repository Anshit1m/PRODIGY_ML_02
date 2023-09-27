import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Mall_Customers.csv')


X = df.iloc[:,[3,4]].values


from sklearn.cluster import KMeans
wcss = []

###elbow meathod
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


###elbow plotting
plt.plot(range(1,11),wcss)
plt.title("Elbow")
plt.xlabel("no. of clusters")
plt.ylabel('wcss values')
plt.show()


### k-means cluster
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
y_kmeans = kmeansmodel.fit_predict(X)

###plotting kmeans
plt.scatter(X[y_kmeans ==0,0], X[y_kmeans == 0,1], s=80, c = "red", label='Customer 1')
plt.scatter(X[y_kmeans ==1,0], X[y_kmeans == 1,1], s=80, c = "cyan", label='Customer 1')
plt.scatter(X[y_kmeans ==2,0], X[y_kmeans == 2,1], s=80, c = "yellow", label='Customer 1')
plt.scatter(X[y_kmeans ==3,0], X[y_kmeans == 3,1], s=80, c = "indigo", label='Customer 1')
plt.scatter(X[y_kmeans ==4,0], X[y_kmeans == 4,1], s=80, c = "green", label='Customer 1')

###centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c = "black",label ='Centroids')
plt.title("Clustersof customers")
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()
