import torch
import torch.nn as nn
from torch.autograd import Variable
import random


class MeanAggregator(nn.Module):

    def __init__(self, features, adj_lists, adj, feat_sims, num_sample=None, cuda=False):

        super(MeanAggregator, self).__init__()

        self.features = features
        self.adj_lists = adj_lists
        self.adj = torch.FloatTensor(adj)
        self.num_nodes = adj.shape[0]
        self.num_sample = num_sample
        self.cuda = cuda
        self.feat_sims = feat_sims

        if cuda:
            self.adj = self.adj.cuda()
            self.feat = self.features(torch.arange(0, self.num_nodes).cuda())
            self.feat_sims = self.feat_sims.cuda()
        else:
            self.feat = self.features(torch.arange(0, self.num_nodes))

    def forward(self, nodes, agg=True):
        if not agg:
            return self.feat[nodes]

        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        _set = set
        if not self.num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        self.num_sample,
                                        )) if len(to_neigh) >= self.num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i].item()]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))

        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        if self.cuda:
            mask = mask.cuda()
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        row = [nodes[i].item() for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        col = [n for samp_neigh in samp_neighs for n in samp_neigh]
        mask[row_indices, column_indices] = 1.0*self.adj[row, col] + 1.0*self.feat_sims[row, col]
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.feat[torch.LongTensor(unique_nodes_list).cuda()]
        else:
            embed_matrix = self.feat[torch.LongTensor(unique_nodes_list)]
        to_feats = mask.mm(embed_matrix)
        return to_feats
