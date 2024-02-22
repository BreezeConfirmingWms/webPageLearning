from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from gcn.utils import load_data, accuracy,download_data,convert_to_webLab
from gcn.models import GCN
from sklearn.metrics import accuracy_score

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,  help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=2022153, help='Random seed.')
parser.add_argument('--n_epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=1024, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
# parser.add_argument('--path', type=str, default='../data/cora/', help='data path')
parser.add_argument('--dataset', type=str, default='WebKB', help='dataset name')
parser.add_argument('--sub_dataset', type=str, default='', help='dataset name')
opt = parser.parse_args()
opt.cuda = not opt.no_cuda and torch.cuda.is_available()


df=pd.DataFrame()
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)


#download_data(opt.dataset)

# 载入数据
adj, features, labels, idx_train, idx_val, idx_test,label_idx = load_data(opt.dataset,opt.sub_dataset)

df["URL"]=label_idx

# CLASS = 7
# EPOCH = 1000
# TEST = 100
# LR = 0.001
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=opt.hidden,
            nclass=labels.max().item() + 1,
            dropout=opt.dropout)

pas=labels.max().item()

criterion=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

if opt.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

#########
#训练 #
#########
t_total = time.time()
for epoch in range(opt.n_epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train =criterion(output[idx_train], labels[idx_train])
    pred,acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not opt.fastmode:
        model.eval()
        output = model(features, adj)

    #loss_val = F.nll_loss(output[idx_val], labels[idx_val])

    loss_val = criterion(output[idx_val], labels[idx_val])
    _,acc_val = accuracy(output[idx_val], labels[idx_val])
    print(f'[Epoch {epoch+1:04d}/{opt.n_epochs}]'
          f'[Train loss: {loss_train.item():.4f}]'
          f'[Train accuracy: {acc_train.item():.4f}]'
          f'[Validation loss: {loss_val.item():.4f}]'
          f'[Validation accuracy: {acc_val.item():.4f}]'
          f'[Time: {time.time() - t:.4f}s]')
    if epoch==opt.n_epochs-1:
        preds=convert_to_webLab(pred)
        df["Label"]=preds

df.to_csv(path_or_buf="../WebPageCls/gnn_data.csv",index=False)  # 将图模型的高精度预训练生成数据写入gnn_data.csv并保存到根目录供机器学习模型提升使用
#print(df.head(5))
#


# ###########
# # 可自行用划分测试子集测试 #
# ###########


# model.eval()
# output = model(features, adj)
# #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
# loss_test = criterion(output[idx_test], labels[idx_test])
# acc_test = accuracy(output[idx_test], labels[idx_test])
# print("[Test result]"
#       f"[Test loss: {loss_test.item():.4f}]"
#       f"[Test accuracy: {acc_test.item():.4f}]")


# model = GCN(features.shape[-1], 1024, CLASS, 0.3, True).cuda()
# # model = GAT(features.shape[-1], 16, CLASS, 0.3, True).cuda()
# loss_fn = nn.CrossEntropyLoss()
# optim = optim.Adam(model.parameters(), lr=LR)
#
# features = features.cuda()
# adj = adj.cuda()
# labels = labels.cuda()
#
# for i in range(EPOCH):
#     optim.zero_grad()
#     outputs = model(features, adj)
#     loss = loss_fn(outputs[0, idx_train], labels[0, idx_train])
#     loss.backward()
#     optim.step()
#
#     # precision
#     predicts = outputs.argmax(dim=2)
#     precision = accuracy_score(labels[0, idx_train].cpu(), predicts[0, idx_train].cpu())
#     print("[{:3d}/{:3d}]  loss: {:.4f}   precision: {:5.2%}".format(i + 1, EPOCH, loss, precision))
#
#     if (i % TEST == TEST - 1):
#         model.eval()
#         outputs = model(features, adj)
#         loss = loss_fn(outputs[0, idx_test], labels[0, idx_test])
#
#         predicts = outputs.argmax(dim=2)
#         precision = accuracy_score(labels[0, idx_test].cpu(), predicts[0, idx_test].cpu())
#
#         print("=============================================")
#         print("[Testing]  loss: {:.4f}   precision: {:5.2%}".format(loss, precision))
#         print("=============================================")
#         model.train()