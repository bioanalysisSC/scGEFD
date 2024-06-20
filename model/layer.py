import warnings

import numpy as np
import torch
import os, sys
import torch.nn as nn
import torch.nn.functional as F
import torch_clustering
import math
import random
import copy
from sklearn.cluster import SpectralClustering
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv, TAGConv
import torch.nn.init as init
from .utils import init_cluster, mask_data, add_noise
from .loss import ZINBLoss, MeanAct, DispAct, KLD


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adjust_learning_rate(optimizer, lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class ClusteringLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.alpha = alpha
        self.clusters = None

    def forward(self, inputs):
        q = 1.0 / (1.0 + (torch.sum((inputs.unsqueeze(1) - self.clusters) ** 2, dim=2) / self.alpha))
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        q = torch.transpose(torch.transpose(q, 0, 1) / torch.sum(q, dim=1), 0, 1)
        return q

class TAGEncoder(nn.Module):

    def __init__(self, num_genes, hidden_dim, latent_dim, dropout=0.4):
        super(TAGEncoder, self).__init__()
        # initialize parameter
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

        self.tag_layer_1 = TAGConv(
            in_channels=num_genes, out_channels=hidden_dim
        )
        self.tag_layer_2 = TAGConv(
            in_channels=hidden_dim, out_channels=latent_dim
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        # Reset tag_layer_1 parameters
        for param in self.tag_layer_1.parameters():
            if len(param.shape) > 1:
                init.xavier_uniform_(param)
            else:
                init.zeros_(param)
        # Reset tag_layer_2 parameters
        for param in self.tag_layer_2.parameters():
            if len(param.shape) > 1:
                init.xavier_uniform_(param)
            else:
                init.zeros_(param)

    def forward(self, x, edge_index):
        x = x.to(torch.float32)
        hidden_out1 = self.tag_layer_1(x, edge_index)
        hidden_out1 = F.leaky_relu(hidden_out1,negative_slope=0.02)
        hidden_out1 = F.dropout(hidden_out1, p=self.dropout, training=self.training)
        hidden_out2 = self.tag_layer_2(hidden_out1, edge_index)
        # hidden_out2 = F.leaky_relu(hidden_out2,negative_slope=0.02)
        hidden_out2 = F.log_softmax(hidden_out2+1e-6)
        
        return hidden_out2


class Bilinear(nn.Module):
    def __init__(self,
                 adj_dim,
                 dropout=0.2,
                 use_bias=False,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):

        super().__init__()
        self.use_bias = use_bias
        self.dropout = dropout
        self.adj_dim = adj_dim
        self.activation = getattr(F, activation) if activation is not None else None

        # Kernel initialization
        if kernel_initializer == 'glorot_uniform':
            self.kernel_initializer = nn.init.xavier_uniform_
        else:
            raise ValueError("Unsupported kernel_initializer:", kernel_initializer)

        # Bias initialization
        if use_bias:
            if bias_initializer == 'zeros':
                self.bias_initializer = nn.init.zeros_
            else:
                raise ValueError("Unsupported bias_initializer:", bias_initializer)
        else:
            self.bias_initializer = None

        self.reset_parameters()

    def reset_parameters(self):
        self.kernel = nn.Parameter(torch.Tensor(self.adj_dim, self.adj_dim))  # Placeholder, adjust shape accordingly
        self.kernel_initializer(self.kernel)

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.adj_dim))  # Placeholder, adjust shape accordingly
            self.bias_initializer(self.bias)
        else:
            self.bias = None

    def forward(self, inputs):
        x = F.dropout(inputs, p=self.dropout)
        h1 = torch.matmul(x, self.kernel)
        output = torch.matmul(h1, torch.transpose(x, 0, 1))

        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output


class SCModel(nn.Module):
    def __init__(self, x, adj, base_encoder, num_genes=2000, n_sample=None, dim=256, hidden_dim=64,
                 adj_dim=32, num_clusters=None):
        super(SCModel, self).__init__()

        self.x = torch.FloatTensor(x).to(device)
        self.adj = adj
        self.dropout = 0.2
        self.num_clusters = num_clusters
        self.n_sample = n_sample

        # create the encoders
        self.encoder = base_encoder(num_genes=num_genes, hidden_dim=dim, latent_dim=hidden_dim)
        self.encoder_mean = nn.Sequential(nn.Linear(hidden_dim, num_genes), MeanAct())
        self.encoder_disp = nn.Sequential(nn.Linear(hidden_dim, num_genes), DispAct())
        self.encoder_pi = nn.Sequential(nn.Linear(hidden_dim, num_genes), nn.Sigmoid())
        self.zinb_loss = ZINBLoss()

        self.bn = nn.BatchNorm1d(hidden_dim, affine=False)

        self.clustering = ClusteringLayer()

        self.decoderA = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, adj_dim),
            Bilinear(adj_dim),
            nn.Sigmoid()
        )

        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.decoderA.children():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.0)
        for layer in self.encoder_mean.children():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.0)
        for layer in self.encoder_disp.children():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.0)
        for layer in self.encoder_pi.children():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0.0)

    def pretrain(self, edge_index, size_factors, pretrain_epochs=1000, lr=1e-4, W_x=1.0, W_a=0.3):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        size_factors = torch.FloatTensor(size_factors).to(device)
        # edge_index = torch.LongTensor(edge_index).to(device)
        edge_index = edge_index.to(device)
        for epoch in range(pretrain_epochs):
            z = self.encoder(self.x, edge_index)
            mean = self.encoder_mean(z)
            disp = self.encoder_disp(z)
            pi = self.encoder_pi(z)
            zinb_loss = self.zinb_loss(x=self.x, mean=mean, disp=disp, pi=pi, scale_factor=size_factors)
            recon_A = self.decoderA(z)
            A_rec_loss = F.mse_loss(self.adj, recon_A)
            loss = W_a * A_rec_loss + W_x * zinb_loss
            optimizer.zero_grad()
            # print("Epoch ", epoch, " loss ", loss.item(), " zinb_loss:", zinb_loss.item(), "  A_rec_loss:", A_rec_loss.item())
            with torch.autograd.detect_anomaly():
                loss.backward()
            # torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=0.5)
            optimizer.step()

            if epoch % 10 == 0:
                print("Epoch ", epoch, " zinb_loss:", zinb_loss.item(), "  A_rec_loss:", A_rec_loss.item())
    
    def off_diagonal(self,x):
        n,m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def fine_tuning(self, adj, adj_sparse, labels, epochs=300, lr=1e-4, info_epoch=5,
                    eval_metric=None, save_path=None, lambd=0.005, cluster_type="Kmeans", adjust=False):

        optimizer = torch.optim.Adam(self.parameters(), lr)
        adj_sparse = adj_sparse.to(device)
        # best_nmi = 0.0
        # best_ari = 0.0
        
        for cur_epoch in range(epochs):
            features = self.encoder(self.x, adj_sparse)
            features = nn.functional.normalize(features,dim=1)
            if cur_epoch == 0:
                psedo_labels, cluster_centers = init_cluster(features, adj.cpu().numpy(), self.num_clusters, labels, cluster_type)
                self.clustering.clusters = cluster_centers.to(device)
                cluster_labels = self.clustering(features.detach())
                pred_labels = cluster_labels.argmax(1)
                results = torch_clustering.evaluate_clustering(labels,
                                                               pred_labels.cpu().numpy(),
                                                               eval_metric=eval_metric,
                                                               phase='ema_train')
                print("###########################")
                print("init results: ", results)
                print("###########################")
            
            if adjust:
                adjust_learning_rate(optimizer, lr, cur_epoch, epochs)
            x_copy = copy.deepcopy(self.x)
            x_aug = add_noise(x_copy)
            x_aug = mask_data(x_aug)
            features_aug = self.encoder(x_aug, adj_sparse)
            features_aug = nn.functional.normalize(features_aug,dim=1)

            c = self.bn(features).T @ self.bn(features_aug)
            c.div_(self.n_sample)
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = self.off_diagonal(c).pow_(2).sum()
            loss = on_diag + off_diag * lambd
            # loss = off_diag * lambd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f'Epoch [{cur_epoch + 1}/{epochs}], loss: {loss}, on_diag: {on_diag}, off_diag: {off_diag}')
            
            if cur_epoch % info_epoch == 0:
                cluster_labels = self.clustering(features)
                pred_labels = cluster_labels.argmax(1)
                results = torch_clustering.evaluate_clustering(labels,
                                                               pred_labels.detach().cpu().numpy(),
                                                               eval_metric=eval_metric,
                                                               phase='ema_train')
                print("Epoch ", cur_epoch, " results: ", results)
                if save_path != None:
                    if cur_epoch == 0:
                        with open(save_path, "w") as f:
                            f.writelines("init results: " + str(results) + "\n")
                    else:
                        with open(save_path, "a") as f:
                            f.writelines(str(cur_epoch+1) + "\t" + str(results) + "\n")
        
        cluster_labels = self.clustering(features)
        pred_labels = cluster_labels.argmax(1)
        results = torch_clustering.evaluate_clustering(labels,
                                                        pred_labels.detach().cpu().numpy(),
                                                        eval_metric=eval_metric,
                                                        phase='ema_train')
        self.nmi = results["ema_train_nmi"]
        self.ari = results["ema_train_ari"]
        self.embedding = features
        self.pred = pred_labels.detach().cpu().numpy()
        if save_path != None:
            with open(save_path, "a") as f:
                f.writelines(str(epochs) + "\t" + str(results) + "\n")
        return self