import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from collections import defaultdict, Counter

import warnings
warnings.simplefilter("ignore", UserWarning)


def load_data(data_dir, dataset, normalized=True):
    edge_path = data_dir + '/' + dataset + '/edges.txt'
    edges = np.loadtxt(edge_path, dtype=int)
    num_nodes = len(np.unique(edges.reshape(-1)))
    adj = np.zeros([num_nodes, num_nodes])
    for edge in edges:
        adj[edge[0]][edge[1]] = 1
        adj[edge[1]][edge[0]] = 1
    for i in range(num_nodes):
        adj[i][i] = 1

    adj_lists = defaultdict(set)
    for edge in edges:
        node1, node2 = edge
        adj_lists[node1].add(node2)
        adj_lists[node2].add(node1)

    fea_path = data_dir + '/' + dataset + '/features.txt'
    features = np.loadtxt(fea_path, dtype=np.float32)[:, 1:]
    if normalized:
        features = col_norm(features)

    label_path = data_dir + '/' + dataset + '/group.txt'
    labels = np.loadtxt(label_path, dtype=int)

    print('features shape:', str(features.shape))

    return edges.tolist(), adj_lists, adj, features, labels


def row_norm(x):
    rowmax = np.max(x, 1)
    rowmax[(rowmax == 0).nonzero()] = 1
    x_norm = x/rowmax.reshape(-1, 1)
    return x_norm


def col_norm(x):
    colmax = np.max(x, 0)
    colmax[(colmax == 0).nonzero()] = 1
    x_norm = x / colmax
    return x_norm


def small_trick(y_test, y_pred):
    y_pred_new = np.zeros(y_pred.shape, np.bool)
    sort_index = np.flip(np.argsort(y_pred, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = sum(y_test[i])
        for j in range(num):
            y_pred_new[i][sort_index[i][j]] = True
    return y_pred_new


def multi_label_classification(X, Y, ratio):

    X = preprocessing.normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=ratio, random_state=42)  # default:42

    logreg = LogisticRegression(random_state=1234, solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    #=========train=========
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg), param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)
    # print('Best parameters')
    # print(clf.best_params_)

    #=========test=========
    y_pred = clf.predict_proba(X_test)
    y_pred = small_trick(y_test, y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    # print("micro_f1: %.4f" % (micro))
    # print("macro_f1: %.4f" % (macro))

    return micro, macro


def cluster(embs, labels):
    num_classes = len(np.unique(labels[:, 1]))
    id2label = dict(zip(labels[:,0], labels[:, 1]))
    label2size = Counter(labels[:, 1])
    y_pred = KMeans(n_clusters=num_classes, random_state=1234).fit_predict(embs)
    label2group = defaultdict(list)
    for i in range(len(y_pred)):
        label2group[id2label[i]].append(y_pred[i])
    ranked_labels = sorted(label2size.items(), key=lambda x:x[1], reverse=True)
    correct_nums = 0
    total_groups = list(range(num_classes))
    for i in range(len(ranked_labels)):
        label = ranked_labels[i][0]
        FLAG = True
        while FLAG:
            bins = np.bincount(label2group[label])
            if len(bins) == 0:
                break
            pred_group = np.argmax(bins)
            if pred_group in total_groups:
                correct_nums += sum(label2group[label]==pred_group)
                total_groups.remove(pred_group)
                FLAG = False
            else:
                while pred_group in label2group[label]:
                    label2group[label].remove(pred_group)
    print('node clustering acc:{:.4f}'.format(correct_nums/len(labels)))

