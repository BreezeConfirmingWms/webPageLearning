论文Combining content and link for classification using matrix factorization
无监督机器学习方法的实现
混合网页内容和指向性连接并使用矩阵分解的主特征提取量化进行机器学习分类预测

--theUtils.py 特征矩阵和邻接矩阵生成(ndarray数据类型),返回监督学习标签索引


运行link_content.py即可观察损失函数下降并得到准确率报告如下
accuracy_score:
0.8318181818181818
              precision    recall  f1-score   support

           0       0.94      0.91      0.93        55
           1       0.79      0.61      0.69        31
           2       0.93      0.65      0.76        20
           3       0.00      0.00      0.00        10
           4       0.78      0.97      0.87       104

    accuracy                           0.83       220
   macro avg       0.69      0.63      0.65       220
weighted avg       0.80      0.83      0.81       220
