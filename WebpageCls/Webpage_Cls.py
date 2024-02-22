

from bagging_randomforest import RandomForest
"""
bagging_randomforest为自己根据基本原理编写的随机森林库，可以调用并进行训练和预测
"""
from xgboost import XGBClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
from FetchData import label_tags
import seaborn as sns
import nltk
import re
from itertools import product
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder






sns.set(font_scale = 1.2, style = "whitegrid")
# 为 seaborn 设置绘图配置

label_dict={
    label_tags[0]:0,
    label_tags[1]:1,
    label_tags[2]:2,
    label_tags[3]:3,
    label_tags[4]:4,
    label_tags[5]:5,
    label_tags[6]:6
}

lemma = WordNetLemmatizer()
# 关于 lemmatization的初始对象创建

df = pd.read_csv(filepath_or_buffer ="data.csv")

df.dropna(how = "any", inplace = True)
# # drop NA's
# df.head(n = 10)
# #print(df.head(n = 10))


def add_url_to_text(data):
    """
     此功能有助于添加关键词
       从 URL 到最终文本
       分析。

      Arguments:

      1. data:数据框中的输入行。

      Returns:

      1. data: 添加了 URL 的结果。
      """

    #data["URL"] = data["URL"].lower().replace(".html", "")
    # 可选择忽略HTML格式


    data["URL"] = re.sub("[\d-]", "", data["URL"])
    # 从文本中删除所有数字
    data["URL"] = re.sub("_",":",data["URL"])
    data["URL"] = data["URL"].replace("^","/",20)
    url_keys_lst = data["URL"].split("/")[3:]
    # 从 URL 中获取关键字
    url_keys_lst = map(lambda x: lemma.lemmatize(x.replace("cs", "computer science")), url_keys_lst)
    # 词形还原关键字
    data["Text"] = "{1} {0}".format(data["Text"], " ".join(url_keys_lst))
    # 在文本中添加关键字
    return data


df = df.apply(func = add_url_to_text, axis = 1)


label_group_data = df.groupby(by = ["Label"]).agg("count").sort_values(by = "URL")
# 按网页真实类别分组
univ_group_data = df.groupby(by = ["University"]).agg("count").sort_values(by = "URL")
# 按大学分组
df["Length"] = df["Text"].apply(lambda x: len(x))


def add_n_grams(data, n):
    """
      此功能有助于获取
       来自输入文本的 n-gram 数据。

      Arguments

      1. data: 输入文本数据来自哪个
       n-gram 将被提取。

      2. n:第'n'个属性的n-gram特征

      Returns

      1. n-gram 文本数据。
      """

    n_gram_str = ""
    # empty string
    data_ngram = nltk.ngrams(data.split(), n=n)
    # generate n-grams
    data_ngram = ["_".join(gram) for gram in data_ngram]
    # iterate over each n-gram
    return " ".join(data_ngram)


df["bigram_text"] = df["Text"].apply(lambda x: add_n_grams(x, 2))
# get bi-gram data
df["trigram_text"] = df["Text"].apply(lambda x: add_n_grams(x, 3))
# get tri-gram data


def feature_extraction_and_scaling(train_x, test_x):
    """
       此功能有助于执行
         特征提取，并缩放数据。

        Arguments:

        1. train_x: 训练数据.

        2. test_x:检测数据.

        Return:

        1.精炼数据集.
        """

    count_vect = CountVectorizer()
    # 初始化count vectors
    tfidf = TfidfTransformer()
    # 初始化 tf-idf 编制对象
    train_x = count_vect.fit_transform(train_x)
    # 将计数向量拟合到数据
    train_x = tfidf.fit_transform(train_x)
    # 训练数据的tf-idf
    test_x = count_vect.transform(test_x)
    test_x = tfidf.transform(test_x)
    # 编制转化测试数据
    scaler = preprocessing.StandardScaler(with_mean=False).fit(train_x)
    # 精练数据集
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    return train_x, test_x

df["features"] = df[['Text', 'bigram_text']].apply(lambda x: ' '.join(x), axis=1)
df["True_type"]=df["Label"]
df["Label"] = LabelEncoder().fit_transform(df["Label"])

setdown="rf"
print("Would you like to select training?")
print("Please input yes or no")
collect=input()
if collect == "yes":
    print("Please select the model from{svm1,svm2,xgb,rf,bayes,knn}")
    setdown=input()


print("Would you like to specify the school website to visited?")
print("Please input yes or no")
collect = input()
if collect == "yes":
    print("please select your destination from:{ cornell,misc,texas,wisconsin,washington}")
    collect = input()
    test = df[df["University"] == collect]
    # test data
    train = df.drop(test.index)
    # train data
    if setdown != "svm1" and setdown != "svm2":
        train_X, train_y, test_X, test_y = train["features"], train["Label"].values, test["features"], test["Label"].values
    else:
        train_X, train_y, test_X, test_y = train["Text"], train["Label"].values, test["Text"], test[
            "Label"].values
else :
    train, test, train_y, test_y = train_test_split(df, df["Label"], test_size=.2, stratify=df["Label"])
    if setdown != "svm1" and setdown != "svm2":
        train_X = train["features"]
        test_X = test["features"]
    else :
        train_X = train["Text"]
        test_X = test["Text"]


train_X, test_X = feature_extraction_and_scaling(train_X, test_X)

print('model is loading...')

top_k_clf = SelectKBest(score_func=f_classif, k=500)
# 选择top——k 特征
train_X = top_k_clf.fit_transform(train_X, train_y)
# 拟合数据

test_X = top_k_clf.transform(test_X)

train_lenghts = train["Length"].values.reshape(len(train["Length"]), 1)
test_lengths = test["Length"].values.reshape(len(test["Length"]), 1)
train_X = np.hstack((train_X.toarray(), train_lenghts))  #将关键语义词组的长度也作为一个特征
test_X = np.hstack((test_X.toarray(), test_lengths))

tray=np.array(train_y)

if setdown=="rf":  ####随机森林算法
    clf = RandomForestClassifier(n_estimators=100)
    # clf=RandomForest(n_estimators=25)

if setdown=="xgb": ###极度梯度提升树
    clf=XGBClassifier()
    # clf=XGBoost()

if setdown=="svm1":   ###非线性SVM
    clf=SVC(kernel="rbf")


if setdown=="svm2":  ####线性SVM
    clf=SVC(kernel="linear")

if  setdown=="bayes":  ####朴素贝叶斯
    clf=MultinomialNB()
if setdown=="knn":   ##### K近邻算法
    clf=KNeighborsClassifier()


clf.fit(train_X,train_y)
predicted=clf.predict(test_X)



accuracy_score = metrics.accuracy_score(test_y, predicted)
print("Accuracy score: {0}".format(round(accuracy_score, 3)))

print(metrics.classification_report(test_y, predicted,target_names=[label_tags[0], label_tags[1],
                                                                    label_tags[2],label_tags[3],
                                                                    label_tags[4],label_tags[5],label_tags[6]
                                                                    ]))

print("model loading complete!")

print("please select the type of web you like to visit")

print("from{course,department,faculty,other,project,staff,student}")

collect = input()

nums=label_dict[collect]
test["pred"]=predicted

print("If you choose to use gnn to optimize ans?")
print("Please input yes or no")
collect = input()

flg=0
if collect == "yes":

    dfs = pd.read_csv(filepath_or_buffer="gnn_data.csv")
    for i in range(len(dfs)):
        temp=dfs.iloc[i,:]
        df_str=temp["URL"]
        df_lab=temp["Label"]
        tst_str=np.array(test["URL"])
        if df_str in tst_str:
            flg+=1
            par1=np.array(test.index[test["URL"] == df_str])
            vals=np.array(test[test["URL"] == df_str]["pred"])
            if vals == df_lab:
                flg-=1
            test.loc[par1,"pred"] =df_lab
            #print(np.array(test[test["URL"] == df_str]["pred"]))
final_pred=np.array(test["pred"])
if (final_pred==predicted).all():
    print("Yes")
else :
    print("No")
print(flg)

accuracy_score1 = metrics.accuracy_score(test_y, final_pred)
print("Accuracy score is optimized from {} to {}".format(round(accuracy_score, 3),round(accuracy_score1,3)))
print(metrics.classification_report(test_y, final_pred,target_names=[label_tags[0], label_tags[1],
                                                                    label_tags[2],label_tags[3],
                                                                    label_tags[4],label_tags[5],label_tags[6]
                                                                    ]))

recommend=test[test["pred"]==nums]

print(recommend.head(n=5)["URL"])


print("The actual ans of search  is below:")

print(recommend.head(n=5)["True_type"])


conf_mat = metrics.confusion_matrix(test_y, final_pred)
conf_mat = pd.DataFrame(conf_mat)
conf_mat.columns = ["course", "department", "faculty", "other", "project", "staff", "student"]
conf_mat.index = conf_mat.columns
# 得到混合矩阵
plt.figure(figsize = (12, 6))
sns.heatmap(conf_mat, robust = True, annot = True, fmt="d", cbar = False, linewidths=.5, cmap="GnBu")
plt.show()