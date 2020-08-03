import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
from networkx.readwrite import json_graph


def load_data(data_dir, dataset, normalized=False):
    with open(data_dir + '/' + dataset + '/' + dataset + '-id_map.json') as f:
        node2id = json.load(f) 

    G_data = json.load(open(data_dir + '/' + dataset + '/' + dataset + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    node2id = {conversion(k):int(v) for k,v in node2id.items()}

    train_ids = []
    val_ids = []
    test_ids = []
    for n in G.nodes():
        if G.node[n]['val']:
            val_ids.append(n)
        elif G.node[n]['test']:
            test_ids.append(n)
        else:
            train_ids.append(n)

    if dataset == 'reddit':
        train_ids = [node2id[n] for n in train_ids]
        test_ids = [node2id[n] for n in test_ids]
        val_ids = [node2id[n] for n in val_ids]

    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print('Removed {:d} nodes that lacked proper annotations'.format(broken_count))    
    for edge in G.edges():
        if G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or G.node[edge[0]]['test'] or G.node[edge[1]]['test']:
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False 

    print('train:%d, test:%d, val:%d' % (len(train_ids), len(test_ids), len(val_ids)))
    
    features = np.load(data_dir + '/' + dataset + '/' + dataset + '-feats.npy')
    if normalized:
        print('normalize...')
        scaler = StandardScaler()
        scaler.fit(features[train_ids])
        features = scaler.transform(features)

    with open(data_dir + '/' + dataset + '/' + dataset + '-class_map.json') as f:
        node2label = json.load(f)
    if dataset == 'ppi':
        labels = {int(i):label for i, label in node2label.items()}
    if dataset == 'reddit':
        labels = {node2id[n]:label for n, label in node2label.items()} 

    return G, node2id, features, labels, train_ids, val_ids, test_ids


def row_norm(x):
    rowmax = np.sum(x, 1)
    rowmax[(rowmax == 0).nonzero()] = 1
    x_norm = x/rowmax.reshape(-1, 1)
    return x_norm


def col_norm(x):
    colmax = np.abs(np.max(np.abs(x), 0))
    colmax[(colmax == 0).nonzero()] = 1
    x_norm = x / colmax
    return x_norm
