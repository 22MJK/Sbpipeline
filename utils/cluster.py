import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def cluster(embeddings_np, eps=0.4, min_samples=2, method="dbscan", distance_metric="cosine"):
    n_samples = embeddings_np.shape[0]
    if n_samples == 0:
        raise ValueError("embeddings_np is empty!")
    # calculate distance matrix
    dist_matrix = pairwise_distances(embeddings_np, metric=distance_metric)
    
    # clustering
    if method.lower() == "dbscan":
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clustering.fit_predict(dist_matrix)
    elif method.lower() == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(distance_threshold=eps, n_clusters=None,
                                             affinity="precomputed", linkage="average")
        labels = clustering.fit_predict(dist_matrix)
    else:
        raise ValueError("method must be 'dbscan' or 'agglomerative'")
    
    
    return labels
def plot_cluster(embeddings_np,labels,basename):
    
    if embeddings_np.shape[1]>2:
        emb_2d = PCA(n_components = 2).fit_transform(embeddings_np)
    else:
        emb_2d = embeddings_np
    
    savedir = f"pics/cluster/{basename}_cluster_result.png"
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=labels, cmap='tab10', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title("Speaker Embeddings Clustering Visualization")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig(savedir,dpi=100)
    plt.close()
