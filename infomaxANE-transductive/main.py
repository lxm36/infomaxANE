import argparse
import time
import torch
import torch.nn as nn
import random
import numpy as np
from utils import load_data, multi_label_classification, cluster
from infomaxANE import infomaxANE

# torch.cuda.set_device(0)


def args_parser():
    parser = argparse.ArgumentParser(description='run infomaxANE in transductive setting')

    # data setting
    parser.add_argument('--data_dir', default='../data/')
    parser.add_argument('--dataset', default='cora', help='cora, citeseer, wiki or pubmed')
    parser.add_argument('--normalized', default=False, type=bool, help='normalized the input features or not')
    parser.add_argument('--order', default=2, type=int,
                        help='aggregate the i-th order surrounding nodes for each node, default=2')
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--savedir', default='../embs')
    parser.add_argument('--neg_order', default=5, type=int,
                        help='negative sampling is performed outside n-hops of the target nodes')

    # model setting
    parser.add_argument('--dim', default=128, type=int, help='number of embedding dimensions, default=128')
    parser.add_argument('--epoch', default=100, type=int, help='default=100')
    parser.add_argument('--print_epoch', default=10, type=int)
    parser.add_argument('--lr', default=0.7, type=float, help='default=0.7')
    parser.add_argument('--num_negs', default=5, type=int, help='number of negative samples generated for each edge')
    parser.add_argument('--batch_size', default=256, type=int, help='default=256')
    parser.add_argument('--clips', default=8, type=int, help='number of local clips for each node, default=8')
    parser.add_argument('--alpha', default=1.0, type=float, help='default=1.0')
    parser.add_argument('--beta', default=1.0, type=float, help='default=1.0')
    parser.add_argument('--gamma', default=1.0, type=float, help='default=1.0')
    parser.add_argument('--seed', default=1, type=int)

    # device setting
    parser.add_argument('--cuda', default=False, type=bool, help='cuda available or not')

    return parser.parse_args()


def evaluate(train_emb, labels):
    '''
    the avaluation method comes from DANE
    '''
    num_classes = len(np.unique(labels[:, 1]))
    train_label = np.zeros((train_emb.shape[0], num_classes), dtype=np.int32)
    for idx, y in labels:
        train_label[idx][y] = 1

    train_ratio = [0.01, 0.05, 0.1, 0.3, 0.5]
    res = []
    for tr in train_ratio:
        test_ratio = 1 - tr
        micro, macro = multi_label_classification(train_emb, train_label, test_ratio)
        res.append('{:.4f}'.format(micro) + ' & ' + '{:.4f} | '.format(macro))
    print(' & '.join(res))


def print_params(mdl):
    total_params = sum(param.numel() for param in mdl.parameters() if param.requires_grad)
    print('total params nums:', str(total_params))


def run(args):
    edges, adj_lists, adj_array, features_array, labels = load_data(args.data_dir, args.dataset, normalized=args.normalized)
    features = nn.Embedding(features_array.shape[0], features_array.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(features_array), requires_grad=False)

    if args.cuda:
        features = features.cuda()
    
    model = infomaxANE(adj_lists, adj_array, features, args)
    print_params(model)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    if args.cuda:
        model.cuda()

    batch_nums = len(edges) // args.batch_size + 1
    total_time = 0
    for epoch in range(args.epoch):
        start_time = time.time()
        random.shuffle(edges)
        epoch_loss = 0

        for batch in range(batch_nums):
            batch_edges = edges[batch*args.batch_size: (batch+1)*args.batch_size]
            optimizer.zero_grad()
            loss = model(batch_edges)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_time = time.time()-start_time
        total_time += epoch_time
        print('epoch:%d, loss:%.6f, time:%.2f' % (epoch, epoch_loss, epoch_time))
        if (epoch+1) % args.print_epoch == 0:
            emb = model.get_all_embs()
            if args.cuda:
                emb = emb.cpu()
            evaluate(emb, labels)
            cluster(emb, labels)
            if args.save:
                np.save(args.savedir + '/' + args.dataset + '/infomaxANE_' + args.dataset + '.npy', emb)
    print('time_per_epoch:%.2f' % (total_time/args.epoch))


if __name__ == '__main__':
    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    cuda_gpu = torch.cuda.is_available()
    if cuda_gpu:
        print('cuda is available!')
    run(args)
