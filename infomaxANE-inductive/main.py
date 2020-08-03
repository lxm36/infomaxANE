import argparse
import time
import torch
import torch.nn as nn
import random
import numpy as np
from utils import *
from infomaxANE import infomaxANE

# torch.cuda.set_device(0)


def args_parser():
    parser = argparse.ArgumentParser(description='run my model')

    # data setting
    parser.add_argument('--data_dir', default='../data/')
    parser.add_argument('--dataset', default='cora', help='cora, citeseer, wiki or pubmed')
    parser.add_argument('--normalized', default=False, type=bool, help='normalized the input features or not')
    parser.add_argument('--order', default=2, type=int, help='aggregate the i-th order surrounding nodes for each node, default=2')
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--savedir', default='./embs')
    parser.add_argument('--neg_order', default=5, type=int)

    # model setting
    parser.add_argument('--dim', default=128, type=int, help='number of embedding dimensions, default=128')
    parser.add_argument('--epoch', default=1, type=int, help='default=100')
    parser.add_argument('--total_batches', default=10**10, type=int)
    parser.add_argument('--eval_batch', default=1000, type=int)
    parser.add_argument('--max_degree', default=100, type=int)
    parser.add_argument('--lr', default=0.7, type=float, help='default=0.7')
    parser.add_argument('--agg_sample1', default=None, type=int)
    parser.add_argument('--agg_sample2', default=None, type=int)
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


def eval_reddit(train_embs, train_labels, test_embs, test_labels):
    print("evaluating with the same method in SAGE")
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score

    log = SGDClassifier(loss="log", random_state=1234, n_jobs=16)
    log.fit(train_embs, train_labels)
    print("Test scores")
    print(f1_score(test_labels, log.predict(test_embs), average="micro"))
    print("Train scores")
    print(f1_score(train_labels, log.predict(train_embs), average="micro"))


def eval_ppi(train_embs, train_labels, test_embs, test_labels):
    print("evaluating with the same method in SAGE")
    np.random.seed(1)
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    scaler = StandardScaler()
    scaler.fit(train_embs)
    train_embs = scaler.transform(train_embs)
    test_embs = scaler.transform(test_embs)
    
    log = MultiOutputClassifier(SGDClassifier(loss="log", random_state=1234), n_jobs=12)
    log.fit(train_embs, train_labels)

    pred_labels = log.predict(test_embs)
    print(f1_score(test_labels, pred_labels, average='micro'))
    print(f1_score(train_labels, log.predict(train_embs), average='micro'))


def print_params(mdl):
    total_params = sum(param.numel() for param in mdl.parameters() if param.requires_grad)
    print('total params nums:', str(total_params))


def run(args):
    G, node2idx, features_array, labels, train_ids, val_ids, test_ids = load_data(args.data_dir, args.dataset, normalized=args.normalized)
    adj, deg = construct_adj(G, node2idx, max_degree=args.max_degree)
    test_adj = construct_test_adj(G, node2idx, max_degree=args.max_degree)

    if not features_array is None:
        # pad with dummy zero vector
        features_array = np.vstack([features_array, np.zeros((features_array.shape[1],))]) 
    features = nn.Embedding(features_array.shape[0], features_array.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(features_array), requires_grad=False)

    adj = torch.LongTensor(adj)
    test_adj = torch.LongTensor(test_adj)
    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
        test_adj = test_adj.cuda()
    model = infomaxANE(adj, test_adj, deg, features, args)
    print_params(model)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    if args.cuda:
        model.cuda()
 
    if args.dataset == 'reddit':
        edges = np.array([[node2idx[edge[0]], node2idx[edge[1]]] for edge in G.edges() if not G[edge[0]][edge[1]]['train_removed']])
    elif args.dataset == 'ppi':
        edges = np.array([[edge[0], edge[1]] for edge in G.edges() if not G[edge[0]][edge[1]]['train_removed']])

    batch_nums = len(edges) // args.batch_size + 1
    batch_count = 0

    for epoch in range(args.epoch):
        np.random.shuffle(edges)
        epoch_loss = 0
        start_time = time.time()
        batch_loss = 0
        for batch in range(batch_nums):
            batch_edges = edges[batch*args.batch_size: (batch+1)*args.batch_size]
            optimizer.zero_grad()
            loss = model(batch_edges)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            epoch_loss += batch_loss

            if (batch_count+1) % args.eval_batch == 0:
                batch_time = time.time()-start_time
                print('epoch:%d, batch:%d, loss:%.6f, time:%.2f' % (epoch, batch_count, batch_loss, batch_time))
                batch_loss = 0
                start_time = time.time()
                evaluate(model, labels, train_ids, val_ids, test_ids, args, flag='val')  # validation
            batch_count += 1

            if batch_count >= args.total_batches:
                return
        print('-----------------epoch:%d--------------------' % epoch)
        print('epoch:%d, epoch_loss:%.6f' % (epoch, epoch_loss))
        evaluate(model, labels, train_ids, val_ids, test_ids, args, flag='val')
        
       
def evaluate(model, labels, train_ids, val_ids, test_ids, args, flag='val'):
    if flag == 'test':
        print('testing...')
    else:
        print('validation...')
    train_embs, val_embs, test_embs = model.get_all_embs(train_ids, test_ids)
    if args.cuda:
        train_embs = train_embs.cpu()
        test_embs = test_embs.cpu()
    np.save(args.savedir + '/infomaxANE_' + args.dataset + '_train_embs.npy', train_embs)
    np.save(args.savedir + '/infomaxANE_' + args.dataset + '_test_embs.npy', test_embs)
    train_labels = np.array([labels[i] for i in train_ids])
    val_labels = np.array([labels[i] for i in val_ids])
    test_labels = np.array([labels[i] for i in test_ids])
    if flag == 'val':
        if args.dataset == 'ppi':
            eval_ppi(train_embs, train_labels, val_embs, val_labels)
        elif args.dataset == 'reddit':
            eval_reddit(train_embs, train_labels, val_embs, val_labels)
    elif flag == 'test':
        if args.dataset == 'ppi':
            eval_ppi(train_embs, train_labels, test_embs, test_labels)
        elif args.dataset == 'reddit':
            eval_reddit(train_embs, train_labels, test_embs, test_labels)


if __name__ == '__main__':
    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    cuda_gpu = torch.cuda.is_available()
    if cuda_gpu:
        print('cuda is available!')
    run(args)
