# Practice-of-Machine-Learning-for-Customer-Clustering
This is for dear Prof. JC's workshop, where we look into AI market segmentation. And we did a machine-learning project, using K-means and Hierarchical clustering(both Agglomerative and Divisive) to create customer clustering for given dataset.

This is a group work of Harry, Elias, Aria and Vincent. Also thanks to Github copilot for some picture generation and polishing.

The general rationale would be: We know K-means method very well and we would like to compare it with hierarchical clustering. However, there is innate difference between these two methods. Hence we apply both midpoint and one-hot encoding for both methods, also both using Euclidean distances, trying to keep them on an 'apple-to-apple' scale.

We noticed some abnormalies in our dataset, including negative INCOME and weird behaviors when TRB(Total Relation Balance)=0. We did not do any changes though because we believe doing selection before clustering is illegal. However, we will introduce Interquartile Range to exclude outliers affecting the scatterplots of clustering results.
(PS: I cannot provide dataset in this project for some reasons)

K-means:

We used original data and IQR-applied data, with Z-scaler for continuous variables, to do K-means clustering. We plotted elbow plot and silhouette plot for selection of clusters, mostly using midpoint method(for there is innate flaw with One-hot-encoding, assuming all values have same distance). After that, we use log1p scaler to test the robustness. But unfortunately this seemed to create new outliers(some TRB values are too small that log(TRB) is -inf).

Hierarchical Clustering:

This is basically mimicing K-means constraints: applying both midpoint and one-hot-encoding. We use ward instead of average for distance measure because it has similar logic to Euclidean distance optimization. Also silhouetter plots are generated to help pick clusters.

At the end of the project, we also provide details for each clusters, like binary percentage map to provide customer portraits.
