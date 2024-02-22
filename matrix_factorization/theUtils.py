import numpy as np
import scipy.sparse as sp
'''
import torch
'''

def encode_onehot(labels):
    label_tag=[]
    labels=np.array(labels,dtype=str)
    classes = np.unique(labels)
    Strclass=[str(s) for s in classes]

    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    for s in labels :
        label_tag.append(Strclass.index(s))

    return label_tag,labels_onehot  # 返回标签和独热编码


def load_data(path="../data/WebKB/", dataset="WebKB"):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels,label_hot = encode_onehot(idx_features_labels[:, -1])
    labels = np.array(labels)
    #建图
    #idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j : i for i, j in enumerate(idx)}

    #edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.dtype(str))

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # 构建对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)

    idx_train = range(int(len(idx_map)*0.8))  # 子集全部用于预训练
    idx_val = range(int(len(idx_map) * 0.6), int(len(idx_map) * 0.9))
    idx_test = range(int(len(idx_map) * 0.8), int(len(idx_map)))

    '''
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    '''

    return adj, features,labels, idx_train, idx_val, idx_test,label_hot


def normalize_adj(mx):
    """归一化稀疏邻接矩阵"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """归一化稀疏特征矩阵"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)