# Machine Learning Project Library


Card dataset for CNN
- https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification

Results:
- Trained with 5 epochs, batch size of 32
- classifies based on rank
- xxx is the joker class

![Confusion Matrix](images/Cards_cm.png)

Each class has 20 cards, 4 suits x 5 samples. Joker only has 5 samples.

My model fits the data well with minimal training time.



Star Dataset for KMeans Clustering
- https://www.kaggle.com/datasets/waqi786/stars-dataset

Results

First, I grid searched for best cluster fit

![Elbow graph 1](images/elbow1.png)

I decided I would cluster using 4 clusters

![clustering 1](images/star_cluster_1.png)

As you can see, majority of the points have ended up in one cluster with a few outliers. I wanted a better look into the main cluster so I dropped 3 of the clusters and did the process again

![Elbow graph 2](images/elbow2.png)

I chose 4 clusters again

![clustering 1](images/star_cluster_2.png)

This time I could see a bit better into the main cluster.
