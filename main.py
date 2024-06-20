import random

import torch
import yaml
import argparse, os, sys
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader

from utils.preprocess import *
from utils.graph_function import get_adj, prepare_graphs
from model.layer import SCModel, TAGEncoder
import warnings


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataname", default="pancreas", type=str)
    parser.add_argument("--save_path", default="./result/", type=str)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    args = parser.parse_args()
    config_path = "./config/" + args.dataname + ".yml"
    with open(config_path) as f:
        if hasattr(yaml, 'FullLoader'):
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        else:
            configs = yaml.load(f.read())
    setup_seed(configs["seed"])
    data_path = configs["base_path"]
    x, y= prepro(data_path)
    
    num_cells = x.shape[0]
    num_genes = x.shape[1]
    print("Cell number:", x.shape[0])
    print("Gene number", x.shape[1])
    x = np.ceil(x).astype(int)
    cluster_number = len(np.unique(y))
    print("Cluster number:", cluster_number)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = normalization(adata, copy=True, highly_genes=configs["highly_genes"], size_factors=True, normalize_input=True,
                          logtrans_input=True)

    count = adata.X
    adj = get_adj(count, n_cluster=cluster_number)
    adj = torch.FloatTensor(adj).to(device)
    edgelist = prepare_graphs(adata=adata, n_cluster=cluster_number, graph_type=configs["graph_type"],graph_distance_cutoff_num_stds=configs["graph_distance_cutoff_num_stds"])
    num_nodes = len(adata.X)
    print(f'Number of nodes in graph: {num_nodes}.')
    print(f'The graph has {len(edgelist)} edges.')
    edge_index = np.array(edgelist).astype(int).T
    edge_index = to_undirected(torch.from_numpy(edge_index).to(torch.long), num_nodes)
    print(f'The undirected graph has {edge_index.shape[1]}.')
    adj_sparse = SparseTensor(
        row=torch.tensor(edge_index[0]),  
        col=torch.tensor(edge_index[1]),  
        sparse_sizes=(num_nodes, num_nodes)
    )
    
    model = SCModel(x=count, adj=adj,base_encoder=TAGEncoder, n_sample=num_cells, num_genes=configs["highly_genes"],
                    num_clusters=cluster_number,dim=configs['dim'], hidden_dim=configs['hidden_dim'], adj_dim=configs['adj_dim']).to(device)
    
    print("Pretrain start")
    model.pretrain(adj_sparse, adata.obs.size_factors, pretrain_epochs=configs["pretrain_epochs"])
    print("Fine-Tuning start")
    results = model.fine_tuning(adj, adj_sparse, y, lambd = configs["lambd"], epochs=configs["epochs"],
            lr=configs["learning_rate"], eval_metric=configs["eval_metric"],save_path = configs["save_path"], cluster_type=configs["cluster_type"])
    print("%s clustering results: NMI=%.4f ARI=%.4f" % (args.dataname, results.nmi, results.ari))
