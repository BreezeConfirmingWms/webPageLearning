from sklearn.decomposition import PCA as pca
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import  SVC

from sklearn import metrics
import theUtils


# 论文中的梯度公式
def gradU(U,Z,Z2,A,gamma=0.01):
    U_grad=np.matmul(np.matmul(Z2,U),Z2)-\
        np.matmul(np.matmul(Z.T,A),Z)+gamma*U
    return U_grad

def gradV(V,Z,Z2,C,alpha=1,beta=0.01):
    V_grad=alpha*(np.matmul(V,Z2)-np.matmul(C.T,Z))+beta*V
    return V_grad

def gradZ(Z,A,V,C,ZU1,ZU2,alpha):
    Z_grad=(np.matmul(np.matmul(ZU1,Z.T),ZU2)+\
            np.matmul(np.matmul(ZU2,Z.T),ZU1)-\
            np.matmul(A.T,ZU2)-\
            np.matmul(A,ZU1))+\
            alpha*(np.matmul(np.matmul(Z,V.T),V)-\
                   np.matmul(C,V))
    return Z_grad

# 错误泛化计算
def calculate_error(A,C,U,V,Z,alpha,beta,gamma):
    error = np.linalg.norm(A-np.matmul(np.matmul(Z,U),Z.T))**2+\
        alpha*(np.linalg.norm(C-np.matmul(Z,V.T))**2)+\
        gamma*(np.linalg.norm(U)**2)+\
        beta*(np.linalg.norm(V)**2)
    return error

def link_content_MF(A,C,l,alpha=0.1,beta=0.01,gamma=0.01,iter_num=1000,learning_rate=0.001):
    print("link & content Matrix Factorization...")
    assert type(A)==np.ndarray
    n,_=A.shape
    _,m=C.shape

    # 参数初始化
    U=np.random.randn(l,l)/98.0
    V=np.random.randn(m,l)/98.0
    Z=np.random.randn(n,l)/98.0

    # 错误记录
    err_list = []

    # 梯度下降法迭代求解
    cnt=0
    for t in range(iter_num):
        error = calculate_error(A,C,U,V,Z,alpha,beta,gamma)
        if (error > 1000000000):
            print("exploded!!!")
        err_list.append(error)

        # 共享计算
        Z2=np.matmul(Z.T,Z)
        ZU1=np.matmul(Z,U.T)
        ZU2=np.matmul(Z,U)

        # 梯度计算
        U_grad=gradU(U,Z,Z2,A,gamma)
        V_grad=gradV(V,Z,Z2,C,alpha,beta)
        Z_grad=gradZ(Z,A,V,C,ZU1,ZU2,alpha)

        # 更新参数
        U -= learning_rate*U_grad
        V -= learning_rate*V_grad
        Z -= learning_rate*Z_grad


        cnt+=1
        print("{} period complete".format(cnt))

    return Z,U,V,err_list


if __name__=='__main__':


    adj, features, labels, _,_,_,_= theUtils.load_data()
    A=adj.A
    C=features.A

    Z,U,V,err_list=link_content_MF(A,C,80,alpha=3,beta=0.1,gamma=0.0001,learning_rate=0.01,iter_num=500)

    print("Z(n x l) :\n",Z)
    print("U(l x l) :\n",U)
    print("V(m x l) :\n",V)
    print(err_list[-1])
    err_log=np.log(np.array(err_list))

    plt.plot(err_list)
    plt.show()
    plt.figure(2)
    plt.plot(err_log)
    plt.show()

    train_X,test_X,train_y,test_y=train_test_split(Z,labels,test_size=.25,stratify=labels)

    clf=SVC(kernel='rbf')

    clf.fit(train_X,train_y)


    preds=clf.predict(test_X)

    accuracy_score = metrics.accuracy_score(test_y, preds)
    print(accuracy_score)

    print(metrics.classification_report(test_y,preds))


"""
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

"""