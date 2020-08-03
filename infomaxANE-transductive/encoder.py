import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from aggregator import MeanAggregator


class Encoder(nn.Module):
    def __init__(self, adj_lists, adj, features, args):
        super(Encoder, self).__init__()
        self.args = args
        self.num_nodes = features.weight.shape[0]
        self.feat_dim = features.weight.shape[1]
        feat_sims = features.weight.mm(features.weight.t())
        self.feat = features
        # print(feat_sims)

        self.agg1 = MeanAggregator(features, adj_lists, adj, feat_sims, cuda=args.cuda)
        self.agg2 = MeanAggregator(lambda nodes: self.agg1(nodes), adj_lists, adj, feat_sims, cuda=args.cuda)

        if args.cuda:
            self.local_weight = nn.Parameter(torch.cuda.FloatTensor(
                self.args.clips, self.args.dim, self.feat_dim))
            init.xavier_uniform(self.local_weight)
        else:
            self.local_weight = nn.Parameter(torch.FloatTensor(
                self.args.clips, self.args.dim, self.feat_dim))
            init.xavier_uniform(self.local_weight)

    def get_all_embs(self, nodes):
        if self.args.order == 1:
            feat = self.agg1(nodes)
        elif self.args.order == 2:
            feat = self.agg2(nodes)
        local_out = torch.einsum('ckd,dn->ckn', self.local_weight, feat.t()).permute(2, 0, 1)
        local_embs = F.relu(local_out)
        return torch.max(local_embs, -2)[0]

    def forward(self, nodes):
        nodes_flat = nodes.reshape(-1)
        if self.args.order == 1:
            feat = self.agg1(nodes_flat)
        elif self.args.order == 2:
            feat = self.agg2(nodes_flat)

        local_out = torch.einsum('ckd,dn->ckn', self.local_weight, feat.t()).permute(2, 0, 1)
        local_embs = F.relu(local_out).view(nodes.shape[0], nodes.shape[1], self.args.clips, -1)

        return local_embs


