import numpy as np
import torch.nn as nn
import torch


class NeighSampler(nn.Module):
    def __init__(self, adj, cuda):
        super(NeighSampler, self).__init__()
        self.adj = adj
        self.cuda = cuda

    def forward(self, nodes, num_neighs):
        adj_lists = self.adj[nodes]
        if num_neighs is None:
            return adj_lists 
        '''adj_lists = np.transpose(adj_lists)
        np.random.shuffle(adj_lists)
        adj_lists = np.transpose(adj_lists)[:, :num_neighs]
        '''
        if self.cuda:
            adj_lists = adj_lists.index_select(1, torch.randperm(self.adj.shape[1]).cuda())[:, :num_neighs]
        else:
            adj_lists = adj_lists.index_select(1, torch.randperm(self.adj.shape[1]))[:, :num_neighs]
        return adj_lists
