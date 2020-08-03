import torch
import torch.nn as nn
from sampler import NeighSampler


def row_norm(x):
    rowmax = torch.max(x, -1)[0]
    rowmax[(rowmax == 0).nonzero()] = 1
    x_norm = x/rowmax.reshape(-1, 1)
    return x_norm


class MeanAggregator(nn.Module):

    def __init__(self, features, og_features, adj, num_nodes,  agg_sample=25, cuda=False, last=False):

        super(MeanAggregator, self).__init__()
        
        self.neigh_sampler = NeighSampler(adj, cuda)
        self.num_nodes = num_nodes
        self.agg_sample = agg_sample
        self.cuda = cuda
        self.feat = features
        self.last = last
        self.og_feats = og_features

        '''
        if cuda:
            self.feat = features(torch.arange(0, self.num_nodes).cuda())
        else:
            self.feat = features(torch.arange(0, self.num_nodes))
        '''

    def forward(self, nodes, agg=True, add_sims=True):
        if not agg:
            return self.feat(torch.LongTensor(nodes).cuda())
        samp_neighs = self.neigh_sampler(nodes, self.agg_sample)
        samp_neighs = torch.cat((nodes.reshape(-1, 1), samp_neighs), -1)
        
        neigh_feats = self.feat(samp_neighs.reshape(-1)).reshape(samp_neighs.shape[0], samp_neighs.shape[1], -1)
        if add_sims:
            og_neigh_feats = self.og_feats(samp_neighs.reshape(-1)).reshape(samp_neighs.shape[0], samp_neighs.shape[1], -1)
            sims = 1 + row_norm(torch.einsum('nd,nds->ns', self.og_feats(nodes), og_neigh_feats.permute(0, 2, 1)))
            sims = (sims/torch.sum(sims, -1, keepdim=True)).unsqueeze(-1)
            sims[torch.isinf(sims)] = 1
            agg_feats = torch.sum(sims*neigh_feats, -2)
        else:
            agg_feats = torch.mean(neigh_feats, -2)

        return agg_feats
