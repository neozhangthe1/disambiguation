from sklearn.cluster import AgglomerativeClustering


def clustering(embeddings, num_clusters):
    model = AgglomerativeClustering(n_clusters=num_clusters).fit(embeddings)
    return model.labels_
