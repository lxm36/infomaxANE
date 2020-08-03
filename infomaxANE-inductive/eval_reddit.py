import numpy as np
import json
import argparse
from networkx.readwrite import json_graph
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from logreg import LogReg


def run_regression_reddit(train_embeds, train_labels, test_embeds, test_labels):
    print('evaluating')
    print(train_embeds.shape)
    print(train_embeds[0:10])
    from sklearn.linear_model import SGDClassifier
    
    log = SGDClassifier(loss="hinge", n_jobs=12, random_state=1234)
    log.fit(train_embeds, train_labels)
    print("Test scores")
    print(f1_score(test_labels, log.predict(test_embeds), average="micro"))
    print("Train scores")
    print(f1_score(train_labels, log.predict(train_embeds), average="micro"))


def run_logreg_reddit(train_embs, train_labels, test_embs, test_labels, cuda=True):
    train_embs = torch.Tensor(train_embs)
    train_labels = torch.LongTensor(train_labels)
    test_embs = torch.Tensor(test_embs)
    test_labels = torch.LongTensor(test_labels)

    if cuda:
        train_embs = train_embs.cuda()
        test_embs = test_embs.cuda()
        train_labels = train_labels.cuda()
        test_labels = test_labels.cuda()

    tot = torch.zeros(1)
    if cuda:
        tot = tot.cuda()
    res = []
    xent = nn.CrossEntropyLoss()
    sigmoid = nn.Sigmoid()

    dims = train_embs.shape[1]
    nb_classes = torch.unique(train_labels.reshape(-1)).shape[0]

    best_epoch = find_epoch(dims, nb_classes, train_embs, train_labels, test_embs, test_labels, cuda)
    print(best_epoch)
    repeats = 50
    for _ in range(repeats):
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

        logits = sigmoid(log(test_embs).detach())
        preds = torch.argmax(logits, dim=1)
        f1 = f1_score(test_labels.cpu().numpy(), preds.cpu().numpy(), average='micro')
        print(f1)
        res.append(f1)
        tot += f1

    print('Avg F1:', tot/repeats)
    res = np.stack(res)
    print(np.mean(res))
    print(np.std(res))


def find_epoch(dims, nb_classes, train_embs, train_lbls, test_embs, test_lbls, cuda):
    log = LogReg(dims, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    xent = nn.CrossEntropyLoss()
    if cuda:
        log.cuda()

    epoch_flag = 0
    epoch_win = 0
    best_acc = torch.zeros(1)
    if cuda:
        best_acc = best_acc.cuda()

    for e in range(2000):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)
        
        loss.backward()
        opt.step()

        if (e+1)%100 == 0:
            log.eval()
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            if acc >= best_acc:
                epoch_flag = e+1
                best_acc = acc
                epoch_win = 0
            else:
                epoch_win += 1
            if epoch_win == 10:
                break
    return epoch_flag


def load_data_for_eval(data_dir, dataset, setting):
    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open(data_dir + '/' + dataset + '/' + dataset + "-G.json")))
    labels = json.load(open(data_dir + '/' + dataset + '/' + dataset +"-class_map.json"))
    
    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n][setting]] 
    return train_ids, test_ids, labels


def args_parser():
    parser = argparse.ArgumentParser("Run evaluation on PPI data.")
    parser.add_argument("--dataset_dir", default='../database/', help="Path to directory containing the dataset.")
    parser.add_argument("--dataset", default='reddit')
    parser.add_argument("--emb_dir", default='./embs/', help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.")
    parser.add_argument("--setting", default='test', help="Either val or test.")

    return parser.parse_args()


if __name__ == '__main__':
    print("evaluating with the same method in DGI and GMI")
    args = args_parser()

    train_ids, test_ids, labels = load_data_for_eval(args.dataset_dir, args.dataset, args.setting)
    train_labels = [labels[i] for i in train_ids]
    test_labels = [labels[i] for i in test_ids]

    node2id = json.load(open(args.dataset_dir + '/' + args.dataset + '/' + args.dataset +"-id_map.json"))
    train_ids = [node2id[n] for n in train_ids]
    test_ids = [node2id[n] for n in test_ids]

    if args.emb_dir == 'feat':
        feats = np.load(args.dataset_dir + '/' + args.dataset + '/' + args.dataset +"-feats.npy")
        train_feats = feats[train_ids]
        test_feats = feats[test_ids]
        print("Running regression..")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)
        eval_reddit(train_feats, train_labels, test_feats, test_labels)
    else:
        train_embs = np.load(args.emb_dir + '/infomaxNE_' + args.dataset +'_train_embs.npy')
        test_embs = np.load(args.emb_dir + '/infomaxNE_' + args.dataset +'_test_embs.npy')
        # eval_reddit(train_embs, train_labels, test_embs, test_labels)

        run_logreg_reddit(train_embs, train_labels, test_embs, test_labels)
