import numpy as np
import torch
import math
import torch_clustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def compute_metrics(y_true, y_pred):
    metrics = {}
    metrics["NMI"] = NMI(y_true, y_pred)
    metrics["ARI"] = ARI(y_true, y_pred)
    return metrics


def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return torch.FloatTensor([data[labels == i].mean(0) for i in range(n_clusters)])


def init_cluster(embedding, adj, cluster_number, gt_labels, cluster_type):
    if cluster_type == "spectral":
        embedding = embedding.detach().cpu().numpy()
        gt_labels = gt_labels
        labels = SpectralClustering(n_clusters=cluster_number,affinity="precomputed", assign_labels="discretize",random_state=0).fit_predict(adj)
        centers = computeCentroids(embedding, labels)
        # results = compute_metrics(gt_labels, labels)
    else:
        embedding = embedding.detach()
        kwargs = {
            'metric': 'cosine',
            'distributed': True,
            'random_state': 0,
            'n_clusters': cluster_number,
            'verbose': True
        }
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=400, n_init=20, tol=1e-4, **kwargs)
        # # clustering_model = torch_clustering.PyTorchGaussianMixture(init='k-means++', max_iter=400, tol=1e-4, **kwargs)
        labels = clustering_model.fit_predict(embedding)
        centers = clustering_model.cluster_centers_
        # results = torch_clustering.evaluate_clustering(gt_labels,
        #                                                     labels.cpu().numpy(),
        #                                                     eval_metric=['nmi','ari'],
        #                                                     phase='ema_train')
    
    return labels, centers


def mask_data(x, row_mask_percent=0.5, col_mask_percent=0.2):
    ncell, ngene = x.size()
    num_masker_rows = int(ncell * row_mask_percent)
    masked_row_indices = torch.randperm(ncell)[:num_masker_rows]
    for row_idx in masked_row_indices:
        num_masker_cols = int(ngene * col_mask_percent)
        masked_col_indices = torch.randperm(ngene)[:num_masker_cols]
        x[row_idx, masked_col_indices] = 0.0
    return x

def add_noise(x, noise_scale=0.1, add_percent=0.3):
    num_nodes, num_genes = x.shape
    num_genes_to_noise = int(num_genes * add_percent)
    gene_indices = torch.randperm(num_genes)[:num_genes_to_noise]
    noise = torch.zeros_like(x)
    noise = noise.to(device)
    noise[:, gene_indices] = torch.randn(num_nodes, num_genes_to_noise).to(device) * noise_scale
    
    return x + noise


if __name__ == "__main__":
    x = torch.rand(10,20).to(device)
    noisy_x = add_noise(x)
    print(x)
    print(noisy_x)
    
