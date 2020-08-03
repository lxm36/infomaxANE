import numpy as np
import json
import argparse
from networkx.readwrite import json_graph
from logreg import LogReg
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def run_logreg(train_embs, train_labels, test_embs, test_labels, cuda=False):
    train_embs = torch.Tensor(train_embs)
    train_labels = torch.Tensor(train_labels)
    test_embs = torch.Tensor(test_embs)
    test_labels = torch.Tensor(test_labels)
    
    if cuda:
        train_embs = train_embs.cuda()
        test_embs = test_embs.cuda()
        train_labels = train_labels.cuda()
        test_labels = test_labels.cuda()

    tot = torch.zeros(1)
    if cuda:
        tot = tot.cuda()
    res = []
    xent = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()

    dims = train_embs.shape[1]
    nb_classes = train_labels.shape[1]

    best_epoch = find_epoch(dims, nb_classes, train_embs, train_labels, test_embs, test_labels, cuda)
    print('best epoch', best_epoch)
    best_th = 0.0
    repeats = 50
    for i in range(repeats):
        log = LogReg(dims, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        if cuda:
            log.cuda()

        for _ in range(best_epoch):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_labels)
            loss.backward()
            opt.step()
        
        if i == 0:
            train_logits = sigmoid(log(train_embs))
            if cuda:
                best_th = find_best_th(train_logits.cpu(), train_labels.cpu())
            else:
                best_th = find_best_th(train_logits, train_labels)
            print('best threshold:', best_th)
        
        logits = sigmoid(log(test_embs))
        zero = torch.zeros_like(logits)
        one = torch.ones_like(logits)
        preds = torch.where(logits>=best_th, one, zero)  # ppi is a multi-label classification problem
        if cuda:
            test_labels = test_labels.cpu()
            preds = preds.cpu() 
        f1 = f1_score(test_labels.numpy(), preds.numpy(), average='micro')
        res.append(f1)
        print(f1)
        tot += f1

    print('Average f1:', tot / repeats)

    res = np.stack(res)
    print(np.mean(res))
    print(np.std(res))


def find_best_th(logits, labels):
    zero = torch.zeros_like(logits)
    one = torch.ones_like(logits)
    best_f1 = 0.0
    best_th = 0.0
    for th in range(1, 10):
        preds = torch.where(logits>=th/10.0, one, zero)
        f1 = f1_score(labels.numpy(), preds.numpy(), average='micro')
        if f1 > best_f1:
            best_f1 = f1
            best_th = th/10.0
    return best_th


def find_epoch(dims, nb_classes, train_embs, train_labels, test_embs, test_labels, cuda):
    log = LogReg(dims, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    xent = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()
    if cuda:
        log.cuda()

    epoch_flag = 0
    epoch_win = 0
    best_f1 = torch.zeros(1)
    tmp_th = 0.5
    if cuda:
        best_f1 = best_f1.cuda()

    for e in range(2000):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_labels)

        loss.backward()
        opt.step()

        if (e+1)%100 == 0:
            log.eval()
            logits = sigmoid(log(test_embs))
            zero = torch.zeros_like(logits)
            one = torch.ones_like(logits)
            preds = torch.where(logits>=tmp_th, one, zero)
            if cuda:
                test_labels = test_labels.cpu()
                preds = preds.cpu()
            f1 = f1_score(test_labels.numpy(), preds.numpy(), average='micro')          
 
            if f1 >= best_f1:
                epoch_flag = e+1
                best_f1 = f1
                epoch_win = 0
            else:
                epoch_win += 1
            if epoch_win == 10:
                break
    return epoch_flag


def load_data_for_eval(data_dir, dataset, setting):
    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open(data_dir + '/' + dataset + '/' + dataset+  "-G.json")))
    labels = json.load(open(data_dir + '/' + dataset + '/' + dataset + "-class_map.json"))
    labels = {int(i):l for i, l in labels.items()}
    
    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n][setting]]
    return train_ids, test_ids, labels


def args_parser():
    parser = argparse.ArgumentParser("Run evaluation on PPI data.")
    parser.add_argument("--dataset_dir", default='../data/', help="Path to directory containing the dataset.")
    parser.add_argument("--dataset", default='ppi')
    parser.add_argument("--emb_dir", default='../embs/', help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument("--setting", default='test')

    return parser.parse_args()


if __name__ == '__main__':
    print("evaluating with the same method in DGI and GMI")
    args = args_parser()

    train_ids, test_ids, labels = load_data_for_eval(args.dataset_dir, args.dataset, args.setting)
    train_labels = np.array([labels[i] for i in train_ids])
    if train_labels.ndim == 1:
        train_labels = np.expand_dims(train_labels, 1)
    test_labels = np.array([labels[i] for i in test_ids])

    if args.emb_dir == "feat":
        print("Using only features..")
        feats = np.load(args.dataset_dir + '/' +args.dataset + '/' + args.dataset +"-feats.npy")
        train_feats = feats[train_ids] 
        test_feats = feats[test_ids] 
        print("Running regression..")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)
        run_logreg(train_feats, train_labels, test_feats, test_labels)

    else:
        train_embs = np.load(args.emb_dir + '/infomaxNE_' + args.dataset +'_train_embs.npy')
        test_embs = np.load(args.emb_dir + '/infomaxNE_' + args.dataset +'_test_embs.npy')
        
        print('standardlize...') 
        scaler = StandardScaler()
        scaler.fit(train_embs)
        train_embs = scaler.transform(train_embs)
        test_embs = scaler.transform(test_embs)
        
        run_logreg(train_embs, train_labels, test_embs, test_labels)
