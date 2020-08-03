import torch
import torch.nn as nn
import numpy as np
import random
from encoder import Encoder


class infomaxANE(nn.Module):
    def __init__(self, adj_lists, context_adj, features, args):
        super(infomaxANE, self).__init__()

        self.adj_lists = adj_lists
        self.context_adj = context_adj
        self.features = features
        self.args = args

        self.encode = Encoder(adj_lists, context_adj, features, args)

        self.xent = nn.CrossEntropyLoss()

    def sample(self, edges):
        node1 = []
        node2 = []
        for edge in edges:
            node = edge[0]
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.args.neg_order):
                current = set()
                for outer in frontier:
                    current |= self.adj_lists[int(outer)]
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(list(range(len(self.adj_lists)))) - neighbors
            neg_samples = random.sample(far_nodes, self.args.num_negs) if self.args.num_negs < len(far_nodes) \
                else far_nodes
            node1.extend([[edge[0]]])
            node2.extend([[edge[1]] + neg_samples])
        return np.array(node1), np.array(node2)

    def prepare(self, edges):
        node1, node2 = self.sample(edges)
        local_emb1 = self.encode(node1)
        local_emb2 = self.encode(node2)
        global_scores = torch.sum(torch.max(local_emb1, -2)[0]*torch.max(local_emb2, -2)[0], -1)
        local_scores = torch.mean(torch.sum(local_emb1*local_emb2, -1), -1)
        if self.args.cuda:
            train_labels = torch.LongTensor(np.zeros(len(edges))).cuda()
        else:
            train_labels = torch.LongTensor(np.zeros(len(edges)))
        if self.args.gamma > 0:
            diff_loss = self.diff(local_emb1) + self.diff(local_emb2)
        else:
            diff_loss = 0
        return global_scores, local_scores, diff_loss, train_labels

    def diff(self, local_embs):
        out = torch.einsum('ijkd, ijdn ->ijkn', local_embs, local_embs.permute(0, 1, 3, 2))
        deno = torch.max(out, -1)[0]
        deno[(deno==0).nonzero()] = 1
        out = out/deno.unsqueeze(-2)
        target =  torch.eye(local_embs.shape[-2], local_embs.shape[-2]).unsqueeze(0).unsqueeze(0)
        if self.args.cuda:
            target = target.cuda()
        diff_loss = torch.sum(torch.abs(out - target))
        return diff_loss/(local_embs.shape[0]*local_embs.shape[1])

    def get_all_embs(self):
        local_embeddings = self.encode.get_all_embs(np.arange(self.features.weight.shape[0]))
        return local_embeddings.detach()

    def score(self, edges):
        node1, node2 = edges[:, 0:1], edges[:, 1:]
        global_emb1, local_emb1 = self.encode(node1)
        global_emb2, local_emb2 = self.encode(node2)
        global_scores = torch.sum(global_emb1*global_emb2, -1)
        local_scores = torch.sum(torch.max(local_emb1,-2)[0]*torch.max(local_emb2, -2)[0], -1)
        return self.args.alpha*global_scores+self.args.beta*local_scores

    def forward(self, edges):
        global_scores, local_scores, diff_loss, labels = self.prepare(edges)
        loss = self.args.alpha*self.xent(global_scores, labels) + self.args.beta*self.xent(local_scores, labels) + self.args.gamma*diff_loss
        return loss
