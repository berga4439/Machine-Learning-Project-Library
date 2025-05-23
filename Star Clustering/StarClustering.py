import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv("star_dataset.csv")

star_classes = df["Spectral Class"]

#Standardizing dataset
df = df[["Luminosity (L/Lo)", "Radius (R/Ro)", "Temperature (K)"]]
X = df.copy().to_numpy() * 1.0
std = np.std(X, axis=0)
X /= std

#looking for best cluster amount
inertia = np.zeros(14)
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, n_init=25).fit(X)
    inertia[i-1] = kmeans.inertia_
#plotting "elbow" graph for best fit
plt.plot(np.arange(1, 15), inertia)
plt.show()

#Using elbow graph, determined 4 clusters was ideal
kmeans = KMeans(n_clusters=4, n_init=25)
kmeans.fit(X)

print(kmeans.inertia_)
print(kmeans.cluster_centers_ * std)

#plotting 3D representation of clusters
labels = kmeans.fit_predict(X)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=50)

centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=200, alpha=0.5, marker='d')

ax.set_xlabel('Luminosity')
ax.set_ylabel('Radius')
ax.set_zlabel('Temperature')

plt.show()

#dropping outliers to re cluster biggest cluster
cluster_sizes = pd.Series(labels).value_counts()

largest_cluster = cluster_sizes.idxmax()

X_filtered = X[labels == largest_cluster]

X_filtered -= np.average(X_filtered, axis=0)
X_filtered /= np.std(X_filtered, axis=0)


inertia = np.zeros(14)
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, n_init=25).fit(X_filtered)
    inertia[i-1] = kmeans.inertia_
#plotting "elbow" graph for best fit of new set
plt.plot(np.arange(1, 15), inertia)
plt.show()


kmeans = KMeans(n_clusters=4, n_init=25)
kmeans.fit(X_filtered)
labels = kmeans.fit_predict(X_filtered)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_filtered[:, 0], X_filtered[:, 1], X_filtered[:, 2], c=labels, cmap='viridis', s=50)

centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=200, alpha=0.5, marker='d')

ax.set_xlabel('Luminosity')
ax.set_ylabel('Radius')
ax.set_zlabel('Temperature')

plt.show()