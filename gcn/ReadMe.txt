图神经网络gnn模型其卷积图神经网络模型形式GCN代码

终端运行 python gcn/gcn_main.py --dataset WebKB  即可进入训练并写入
预训练参数文件gnn_data.csv，可以手动添加到WebPageCls文件夹目录下
--layers.py 网络层定义
——
--models.py GCN模型搭建
——
--utils.py 导入数据并生成特征矩阵和邻接矩阵(pytorch Tensor类型）