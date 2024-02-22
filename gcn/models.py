import torch.nn as nn
import torch.nn.functional as F
from gcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        """
        简单的GCN网络搭建
        nfeat: 特征数
        nhid: 隐藏层数
        nclass: 结构类别数
        """
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """

        :param x: 前向特征
        :param adj: 邻接权重取值
        :return: 预测传导张量
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)