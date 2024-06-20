import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10


def get_PPMI_adj(count, n_cluster=10, k=15, pca=50, mode="connectivity", metric="euclidean", s=2):
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count

    N = len(count)
    avg_N = N // n_cluster
    k = avg_N // 10
    k = min(k, 20)
    k = max(k, 6)
    A = kneighbors_graph(countp, k, mode=mode, metric=metric, include_self=True)
    adj = A.toarray()

    node = adj.shape[0]
    A = random_surf(adj, s, 0.98)
    ppmi = PPMI_matrix(A)
    for i in range(node):
        ppmi[i] = ppmi[i] / (np.max(ppmi[i]))
    adj = ppmi
    return adj


def random_surf(cosine_sim_matrix, num_hops, alpha):
    num_nodes = len(cosine_sim_matrix)
    adj_matrix = cosine_sim_matrix
    P0 = np.eye(num_nodes, dtype='float32')
    P = np.eye(num_nodes, dtype='float32')
    A = np.zeros((num_nodes, num_nodes), dtype='float32')

    for i in range(num_hops):
        P = (alpha * np.dot(P, adj_matrix)) + ((1 - alpha) * P0)
        A = A + P

    return A


def PPMI_matrix(A):
    num_nodes = len(A)
    row_sum = np.sum(A, axis=1).reshape(num_nodes, 1)
    col_sum = np.sum(A, axis=0).reshape(1, num_nodes)
    D = np.sum(col_sum)
    PPMI = np.log(np.divide(np.multiply(D, A), np.dot(row_sum, col_sum)))
    PPMI[np.isinf(PPMI)] = 0.0
    PPMI[np.isneginf(PPMI)] = 0.0
    PPMI[PPMI < 0.0] = 0.0

    return PPMI


def symmetric_normalized_adjacency(adj):

    # 计算每个节点的度
    degree = np.sum(adj, axis=1)
    # 计算度矩阵的平方根
    degree_sqrt_inv = np.power(degree, -0.5)
    degree_sqrt_inv[degree_sqrt_inv == float('inf')] = 0
    # 构造对角矩阵
    if sp.issparse(adj):
        D_sqrt_inv = sp.diags(degree_sqrt_inv)
    else:
        D_sqrt_inv = np.diag(degree_sqrt_inv)
    normalized_adj = D_sqrt_inv.dot(adj).dot(D_sqrt_inv)
    return normalized_adj


def correlation(data_numpy, k, corr_type='pearson'):
    df = pd.DataFrame(data_numpy.T)
    corr = df.corr(method=corr_type)
    nlargest = k
    order = np.argsort(-corr.values, axis=1)[:, :nlargest]
    neighbors = np.delete(order, 0, 1)
    return corr, neighbors


def prepare_graphs(adata, n_cluster, graph_type, graph_distance_cutoff_num_stds):

    count = adata.X
    N = len(count)
    k = 15

    if graph_type == 'KNN':
        print('Computing KNN graph by scanpy...')
        distances_csr_matrix = \
            sc.pp.neighbors(adata, n_neighbors = k + 1, knn=True, copy=True, metric='cosine').obsp[
                'distances']
        distances = distances_csr_matrix.A
        # resize
        neighbors = np.resize(distances_csr_matrix.indices, new_shape=(distances.shape[0], k))

    elif graph_type == 'PKNN':
        print('Computing PKNN graph...')
        if isinstance(adata.X, np.ndarray):
            X = adata.X
        else:
            X = adata.X.toarray()
        distances, neighbors = correlation(data_numpy=X, k=k+1)

    if graph_distance_cutoff_num_stds:
        cutoff = np.mean(np.nonzero(distances), axis=None) + float(graph_distance_cutoff_num_stds) * np.std(
            np.nonzero(distances), axis=None)

    edgelist = []
    for i in range(neighbors.shape[0]):
        for j in range(neighbors.shape[1]):
            if neighbors[i][j] != -1:
                pair = (str(i), str(neighbors[i][j]))
                if graph_distance_cutoff_num_stds:
                    distance = distances[i][j]
                    if distance < cutoff:
                        if i != neighbors[i][j]:
                            edgelist.append(pair)
                else:
                    if i != neighbors[i][j]:
                        edgelist.append(pair)

    return edgelist


def get_adj(count, k=15, n_cluster=None, pca=50, mode="connectivity"):

    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    N = len(count)
    avg_N = N // n_cluster
    k = avg_N // 10
    k = min(k, 20)
    k = max(k, 6)
    k = 15
    A = kneighbors_graph(countp, k + 1, mode=mode, metric="cosine", include_self=True)
    adj = A.toarray()
    adj = adj - np.eye(N)
    return adj