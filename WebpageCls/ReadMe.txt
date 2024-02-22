webkb:官网下载原始文件有编码问题，运行ConvertUtf-8.py文件即可，对报错文件名
要手动调节
webkbs 统一UTF-8编码处理后的文件夹，项目直接依赖数据库

FetchData:对webkbs下的文件读取文本预处理数据，两种形式：纯文本和
精简文本

data.csv 文本网址类别信息表格（大） // datar.csv （小）

Rdf:随机森林 // SVM：支持向量机 // Bayes :朴素贝叶斯法//Xgb:极度梯度提升树
Knn: k近邻
Ans文件存储报告结果

Rdfcls//SVMcls//XGboostCls.py  三种主流机器学习分类器对纯文本和
精简文本的测试效果

IO_report交互文件日志，若要利用文本预测结果推送指定类别网站，
运行WebPage_cls.py

全面的数据可视化以及最优随机森林模型预测 运行WebKB_analysis.py文件

gnn_data.csv为GNN模型预训练高精度参数文件，可以提高预测准确率，见交互脚本
WebPage_cls.py

bagging_randomforest.py自行实现的随机森林模型，训练时间长且过拟合
visionData文件夹为可视化图片存储位置