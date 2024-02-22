import numpy as np
import scipy.sparse as sp
import torch
import wget, os, tarfile, sys


def delete_no_feature_node(edges_unordered,nodes):
    """

    :param edges_unordered:  无向边
    :param nodes:   特征结点
    :return: 删除无特征多余结点后的子图
    """
    if not bool(set(edges_unordered.flatten()) - set(nodes)):
        return edges_unordered
    no_feature_nodes = []
    for ii, edge in enumerate(edges_unordered):
        inter = set(edge) - set(nodes)
        if bool(inter):
            no_feature_nodes.append(ii)
    edges_unordered = np.delete(edges_unordered, no_feature_nodes, axis=0)
    return edges_unordered


def encode_onehot(labels):
    """

    :param labels: 原网址一一对应字符串标签
    :return: 数据集编号标签
    {course:0 ,faculty:1, project:2 , staff:3 ,student:4}按字典序
    """
    classes =np.unique(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset="WebKB", sub_dataset=""):
    """
    加载引文网络数据集（仅限 WebKB）
    论文: Combining content and link for classification using matrix factorization
    """
    print('Loading {} dataset...'.format(dataset))
    path = f"../data/{dataset}/"
    if sub_dataset != "":
        dataset=sub_dataset

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # 建图过程
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}

    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
    #                                 dtype=np.int32)
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.dtype(str))

    edges_unordered = delete_no_feature_node(edges_unordered,idx)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # 根据论文方法构建对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(int(len(idx_map))) # 子集全部用于预训练
    idx_val = range(int(len(idx_map) * 0.5), int(len(idx_map) * 0.7))
    idx_test = range(int(len(idx_map) * 0.8), int(len(idx_map)))



    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test,idx


def download_data(dataset='WebKB'):
    # 可以下载WebKB图形模型数据集的子集压缩包到本地解压，再按顺序全部合并到WebKB.cites和WebKB.content文件里
    if os.path.isdir(f'../data/{dataset}'):
        return
    url = f'https://linqs-data.soe.ucsc.edu/public/lbc/{dataset}.tgz'
    path = f'../data/{dataset}.tgz'
    os.makedirs(f'../data/{dataset}',exist_ok=True)
    if not os.path.isfile(path):
        try:
            wget.download(url, '../data')
        except Exception as e:
            print(e)
            sys.exit(0)
    tar = tarfile.open(path, 'r')
    tar.extractall('../data/')
    tar.close()
    os.remove(path)

def normalize(mx):
    """行归一化稀疏矩阵--特征"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """行归一化稀疏矩阵--邻接矩阵"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def accuracy(output, labels):
    #  精确度计算同时返回预测标签作为预训练数据
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    predict=preds.cpu().numpy()
    return predict,correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将 scipy 稀疏矩阵转换为Tensor稀疏张量."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def convert_to_webLab(arr):
    # 编码转译，机器学习文本预测分类代码中的编码
    for i in range(len(arr)):
        if arr[i] == 0 :
            arr[i] = 0
        elif arr[i] == 1:
            arr[i] = 2
        elif arr[i] == 2:
            arr[i] = 4
        elif arr[i] == 3:
            arr[i] = 5
        elif arr[i] == 4:
            arr[i] = 6
    return arr
