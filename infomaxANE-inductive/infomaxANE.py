import torch
import torch.nn as nn
import numpy as np
import random
from encoder import Encoder
from utils import abs_max, fixed_unigram_candidate_sampler


class infomaxANE(nn.Module):
    def __init__(self, adj, test_adj, deg, features, args):
        super(infomaxANE, self).__init__()
        self.adj_lists = adj
        self.deg = deg
        self.features = features
        self.args = args
        
        self.encode = Encoder(adj, test_adj, features, args)
        
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
                    current |= set(self.adj_lists[int(outer)])
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(list(range(len(self.adj_lists)))) - neighbors
            neg_samples = random.sample(far_nodes, self.args.num_negs) if self.args.num_negs < len(far_nodes) \
                else far_nodes
            node1.extend([[edge[0]]])
            node2.extend([[edge[1]] + neg_samples])
        return np.array(node1), np.array(node2)

    def rand_sample(self, edges):
        negs = fixed_unigram_candidate_sampler((len(edges), self.args.num_negs), False, len(self.deg), 0.75, self.deg)
        nodes1 = edges[:, 0:1]
        nodes2 = np.hstack((edges[:, 1:], negs))
        return nodes1, nodes2

    def prepare(self, edges):
        node1, node2 = self.rand_sample(np.array(edges))
        node1 = torch.LongTensor(node1)
        node2 = torch.LongTensor(node2)
        if self.args.cuda:
            node1 = node1.cuda()
            node2 = node2.cuda()

        local_emb1 = self.encode(node1)
        local_emb2 = self.encode(node2)
        global_emb1 = torch.max(local_emb1, -2)[0]
        global_emb2 = torch.max(local_emb2, -2)[0]

        global_scores = torch.sum(global_emb1*global_emb2, -1)
        local_scores = torch.mean(torch.sum(local_emb1*local_emb2, -1), -1)
        train_labels = torch.LongTensor(np.zeros(len(edges)))
        if self.args.cuda:
            train_labels = train_labels.cuda()
        if self.args.gamma > 0:
            constrain_loss = self.constrain(local_emb1) + self.constrain(local_emb2)
        else:
            constrain_loss = 0
        return global_scores, local_scores, constrain_loss, train_labels

    def constrain(self, local_embs):
        out = torch.einsum('ijkd, ijdn ->ijkn', local_embs, local_embs.permute(0, 1, 3, 2))
        deno = torch.max(out, -1)[0]
        deno[(deno==0).nonzero()] = 1
        out = out/deno.unsqueeze(-2)
        target =  torch.eye(local_embs.shape[-2], local_embs.shape[-2]).unsqueeze(0).unsqueeze(0)
        if self.args.cuda:
            target = target.cuda()
        diff_loss = torch.sum(torch.abs(out - target))
        return diff_loss/(local_embs.shape[0]*local_embs.shape[1])

    def get_all_embs(self, train_ids, val_ids, test_ids):
        if self.args.cuda:
            train_ids = torch.LongTensor(train_ids).cuda()
            val_ids = torch.LongTensor(val_ids).cuda()
            test_ids = torch.LongTensor(test_ids).cuda()
        else:
            train_ids = torch.LongTensor(train_ids)
            val_ids = torch.LongTensor(val_ids)
            test_ids = torch.LongTensor(test_ids)
        train_embs  = self.encode.get_final_embs(train_ids)
        val_embs = self.encode.get_final_embs(val_ids)
        test_embs  = self.encode.get_final_embs(test_ids)
        
        return train_embs.detach(), val_embs.detach(), test_embs.detach()

    def forward(self, edges):
        global_scores, local_scores, constrain_loss, labels = self.prepare(edges)
        loss = self.args.alpha*self.xent(global_scores, labels) + self.args.beta*self.xent(local_scores, labels) + self.args.gamma*constrain_loss
        return loss
