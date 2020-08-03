import torch
import torch.nn as nn
from torch.nn import init
from aggregator import MeanAggregator


class Encoder(nn.Module):
    def __init__(self, adj, test_adj,  features, args):
        super(Encoder, self).__init__()
        
        self.args = args
        self.num_nodes = features.weight.shape[0]
        self.feat_dim = features.weight.shape[1]
        self.feat = features
        self.act = nn.ReLU()

        self.agg1 = MeanAggregator(features, features, adj, self.num_nodes, agg_sample=args.agg_sample1, cuda=args.cuda)
        self.agg2 = MeanAggregator(lambda nodes: self.agg1(nodes), features, adj, self.num_nodes, agg_sample=args.agg_sample2, cuda=args.cuda, last=True)

        self.test_agg1 = MeanAggregator(features, features, test_adj, self.num_nodes, agg_sample=args.agg_sample1, cuda=args.cuda)
        self.test_agg2 = MeanAggregator(lambda nodes: self.test_agg1(nodes), features, test_adj, self.num_nodes, agg_sample=args.agg_sample2, cuda=args.cuda, last=True) 

        if args.cuda:
            self.local_weight = nn.Parameter(torch.cuda.FloatTensor(self.args.clips, self.args.dim, self.feat_dim))
            init.xavier_uniform(self.local_weight)
        else:
            self.local_weight = nn.Parameter(torch.FloatTensor(self.args.clips, self.args.dim, self.feat_dim))
            init.xavier_uniform(self.local_weight)

    def get_final_embs(self, nodes):
        batch_size = self.args.batch_size
        batch_num = len(nodes) // batch_size + 1
        local_outs = torch.zeros((len(nodes), self.args.clips, (self.args.order+1)*self.args.dim))
        if self.args.cuda:
            local_outs = local_outs.cuda()
        for i in range(batch_num):
            agg_feats = [] 
            if self.args.order >= 1:
                agg_feats.append(self.test_agg1(nodes[i*batch_size:(i+1)*batch_size]))
            if self.args.order >= 2:
                agg_feats.append(self.test_agg2(nodes[i*batch_size:(i+1)*batch_size]))
            agg_feats.append(self.feat(nodes[i*batch_size:(i+1)*batch_size]))
            part_embs = []
            for feats in agg_feats:
                part_out = torch.einsum('ckd,dn->ckn', self.local_weight, feats.t()).permute(2, 0, 1)
                part_embs.append(part_out)
            local_outs[i*batch_size:(i+1)*batch_size] = torch.cat(part_embs, -1)
        local_embs = self.act(local_outs)
        final_embs = torch.cat((torch.max(local_embs, -2)[0], torch.mean(local_embs, -2)), -1)
        return final_embs  # return concat of the max and the mean of local embs
        # return torch.max(local_embs, -2)[0]  #  only return the max of local_embs

    def forward(self, nodes):
        nodes_flat = nodes.reshape(-1)
        agg_feats = []
        if self.args.order >= 1:
            agg_feats.append(self.agg1(nodes_flat))
        if self.args.order >= 2:
            agg_feats.append(self.agg2(nodes_flat))
        agg_feats.append(self.feat(nodes_flat))
        local_outs = []
        for feats in agg_feats:
            local_out = torch.einsum('ckd,dn->ckn', self.local_weight, feats.t()).permute(2, 0, 1)
            local_outs.append(local_out.view(nodes.shape[0], nodes.shape[1], self.args.clips, -1))
        local_embs = torch.cat(local_outs, -1)
        return self.act(local_embs)


