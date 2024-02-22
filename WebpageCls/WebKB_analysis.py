import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from itertools import product
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from matplotlib.colors import ListedColormap


sns.set(font_scale = 1.2, style = "whitegrid")
# set plotting config for seaborn

lemma = WordNetLemmatizer()
# initialize object for lemmatization

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

    data["URL"] = data["URL"].lower().replace(".html", "")
    #忽略HTML格式
    data["URL"] = re.sub("[\d-]", "", data["URL"])
    # 从文本中删除所有数字
    url_keys_lst = data["URL"].split("^")[3:]
    # 从 URL 中获取关键字
    url_keys_lst = map(lambda x: lemma.lemmatize(x.replace("cs", "computer science")), url_keys_lst)
    # 词形还原关键字
    data["Text"] = "{1} {0}".format(data["Text"], " ".join(url_keys_lst))
    # 向文本中添加关键字
    return data


df = df.apply(func = add_url_to_text, axis = 1)


label_group_data = df.groupby(by = ["Label"]).agg("count").sort_values(by = "URL")
# 按类别分组
univ_group_data = df.groupby(by = ["University"]).agg("count").sort_values(by = "URL")
# 按大学名称分组

def plot_data(df, x, y, title, xlabel=None, ylabel=None, angle=0):
    """
    此功能有助于可视化数据
     分配。

    Arguments:

    1. df: 输入 pandas dataframe.
    2. x: The x-axis 列名或索引.
    3. y: The y-axis  列名或索引.
    4. title: The title of the plot.
    5. xlabel:  x-axis 标签, 默认None.
    6. ylabel:  y-axis 标签,默认 None.
    7. angle: The x-axis 旋转刻度.
    """

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x=x, y=y, ci=None)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(rotation=angle, fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

plot_data(label_group_data, x = label_group_data.index, y = "URL", title = "Data distribution across labels",
         xlabel = "Class labels", ylabel = "Count")

plot_data(univ_group_data, x = univ_group_data.index, y = "URL", title = "Data distribution across Universities",
         xlabel = "University", ylabel = "Count")



df["Length"] = df["Text"].apply(lambda x: len(x))


g = sns.FacetGrid(data = df, col = "Label", col_wrap = 3, hue = "University", palette = "Dark2")
# 创建分面图
g = g.map(plt.plot, "Length").add_legend()
# 绘制数据
g.set(xlabel = "Doc. index", ylabel = "Length")
# 设置轴标签
plt.suptitle("Text length across categories", fontsize = 18)

plt.subplots_adjust(top=0.85)


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

    data_ngram = nltk.ngrams(data.split(), n=n)
    # 生成 n-gram
    data_ngram = ["_".join(gram) for gram in data_ngram]
    # 遍历每个 n-gram
    return " ".join(data_ngram)


df["bigram_text"] = df["Text"].apply(lambda x: add_n_grams(x, 2))
#获取bi-gram数据
df["trigram_text"] = df["Text"].apply(lambda x: add_n_grams(x, 3))
# 获取 tri-gram 数据

def frequency_charts(df, wordcloud=False, top=10, title=None):
    """
    此功能有助于执行频率
     对数据进行分析。

    Arguments:

    1. df: 输入 pandas dataframe.
    2. wordcloud: 如果为真，则绘制一个 wordcloud 地图，分开
     来自共同词情节.
    3. top: 要绘制的前“N”个单词，默认为 10.
    4. title: plot的标题，默认为None.
    """

    word_tokens = []
    # 单词标记的空列表
    for sentence in df:
        # 遍历每个句子
        word_tokens.extend(sentence.split())
        # 添加工作令牌
    text_nltk = nltk.Text(word_tokens)
    # 生成 nltk 文本
    text_freq = nltk.FreqDist(text_nltk)
    # 获取文本频率

    top_words = text_freq.most_common(n=top)
    words_tuple, frequeny_tuple = zip(*top_words)
    # 单词，以及它们的频率
    plot_data(df=None, x=list(words_tuple), y=list(frequeny_tuple), xlabel="Words",
              ylabel="Count", title=title, angle=60)

    if wordcloud:
        # 生成词云
        wordcloud = WordCloud().generate(" ".join(word_tokens))
        plt.figure(figsize=(12, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

#
frequency_charts(df["Text"], title="Common words - Unigram")
#
#
frequency_charts(df["bigram_text"], title = "Common words - Bi-gram")
#
frequency_charts(df["trigram_text"], title = "Common words - Tri-gram")



df["features"] = df[['Text', 'bigram_text']].apply(lambda x: ' '.join(x), axis=1)
# get the feature set for the model


frequency_charts(df["features"], title = "Features", wordcloud = True)


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, validation_curve, learning_curve,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

df["Label"] = LabelEncoder().fit_transform(df["Label"])
# 将文本标签编码为数字
test = df[df["University"] == "wisconsin"]
# 测试数据
train = df.drop(test.index)
# 训练数据

train_X, train_y, test_X, test_y = train["features"], train["Label"].values, test["features"], test["Label"].values




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
    # initialize count vectors
    tfidf = TfidfTransformer()
    # initialize tf-idf transformer
    train_x = count_vect.fit_transform(train_x)
    # fit count vector to data
    train_x = tfidf.fit_transform(train_x)
    # tf-idf on train data
    test_x = count_vect.transform(test_x)
    test_x = tfidf.transform(test_x)
    # transform the test data
    scaler = preprocessing.StandardScaler(with_mean=False).fit(train_x)
    # scaling of data
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    return train_x, test_x

train_X, test_X = feature_extraction_and_scaling(train_X, test_X)

n_est_lst = range(25, 101, 25)
# 随机森林的估计器数量
k_lst = [500, 1000, 2000, 5000, 10000, 100000, 200000, train_X.shape[1]]
# 要尝试的功能数量


def plot_data_2(df, n_est_lst):
    """
   此函数绘制指标数据
     用于调整的各种参数
     使用交叉验证。

    Arguments:

    1. df: 用于绘图的输入数据框。
    2. n_est_lst: 用于随机森林的估计器参数。
    """

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(15, 7))
    ax = ax.flatten()
    # 生成子图
    for i in range(len(n_est_lst)):
        # 迭代每个估计值
        #pos=df["n_estimators"]
        data = df[df["n_estimators"] == n_est_lst[i]].sort_values(by = "k")
        ax[i].plot(data["k"], data["f1"], "r--")
        ax[i].set_title("Num of estimators: " + str(n_est_lst[i]), fontsize=12)
        ax[i].set_xlabel("Features used", fontsize=12)
        ax[i].set_ylabel("f1", fontsize=12)
    plt.tight_layout()
    plt.suptitle("f1-score Vs Number of features for various estimator values", fontsize=15, y=1.02)
    plt.show()


def plot_model_selection(function):
    """
    此功能遵循模型选择过程。
     根据 CV 结果，绘制
     使用的所有参数。

    Argument:

    1. function: 在这种情况下, it's "model_selection_process func"
    """

    def wrapper(*args, **kwargs):
        """
        包装封装函数
        """

        clf = function(*args, **kwargs)
        # 模型选择分类器
        cv_results, results_dict = clf.cv_results_, {}
        # 简历结果和字典
        for i in range(len(cv_results["params"])):
            # iterate over index of parameter
            key = tuple(cv_results["params"][i].items())
            # 字典的钥匙
            results_dict[key] = []
            # 创建密钥
            for j in range(kwargs["num_splits"]):
                # 遍历每个拆分
                score = cv_results["split{0}_test_score".format(j)][i]
                results_dict[key].append(score)
                # 添加分数结果
            results_dict[key] = np.mean(results_dict[key])
            # 取平均结果
        df = pd.DataFrame()
        # 空数据框
        val = list(zip(*results_dict.items()))[0]
        # cpt1=zip(*val)
        # cpt2=list(zip(*val))[0]
        # cpt3=zip(*list(zip(*val))[0])
        df["k"] =list(zip(*list(zip(*val))[1]))[1]
        df["n_estimators"] = list(zip(*list(zip(*val))[0]))[1]
        df["f1"] = results_dict.values()
        #  dataframe 赋值
        plot_data_2(df, kwargs["n_est_lst"])
        return clf

    return wrapper


@plot_model_selection
def model_selection_process(train_x, train_y, k_lst, n_est_lst, num_splits=3):
    """
    此功能有助于执行
     使用交叉验证的模型选择过程。

    Arguments:

    1. train_x: 输入特征。
    2. train_y:输出类标签
    3. k_lst: 要尝试的各种功能
    4. n_est_lst: 随机森林要尝试的估计器数量.

    Returns:

    1. 建立的分类器模型.
    """

    cv = StratifiedShuffleSplit(n_splits=num_splits, test_size=0.1, random_state=0)
    # 定义如何拆分数据以进行交叉验证
    top_k = SelectKBest(score_func=f_classif, k=1)
    # 使用卡方测试获取特征top-k的方法
    pipe = Pipeline([("topk", top_k), ("rforest", RandomForestClassifier())])
    # 定义工作流的管道
    params = {"topk__k": k_lst,"rforest__n_estimators": n_est_lst}
    # 模型选择要测试的参数
    clf = GridSearchCV(pipe, param_grid=params, n_jobs=-1, cv=cv, scoring="f1_weighted")
    # 网格搜索优化模型选择过程，自适应超参数
    clf.fit(train_x, train_y)
    #  拟合数据
    return clf


clf = model_selection_process(train_x = train_X, train_y = train_y, k_lst = k_lst,
                             n_est_lst = n_est_lst, num_splits = 3)

print("Best Parameters")
print(clf.best_params_)


top_k_clf = SelectKBest(score_func = f_classif, k = clf.best_params_["topk__k"])
# 选择top- K 个特征
train_X = top_k_clf.fit_transform(train_X, train_y)
# 拟合数据

test_X = top_k_clf.transform(test_X)

"""
Validation scores  验证集评分机制
"""

param_lst = [25, 50, 75, 100, 125, 150, 175, 200]
# 各种参数值
cv = StratifiedShuffleSplit(n_splits = 3, test_size = 0.1, random_state = 0)
# 定义如何拆分数据以进行交叉验证
train_scores, test_scores = validation_curve(RandomForestClassifier(), train_X, train_y,
                                            param_name = "n_estimators", param_range  = param_lst,
                                            scoring = "f1_weighted", n_jobs = -1, cv = cv)
# 获得不同深度的决策树的验证结果
train_score_mean = np.mean(train_scores, axis = 1)
train_score_std = np.std(train_scores, axis = 1)
test_score_mean = np.mean(test_scores, axis = 1)
test_score_std = np.std(test_scores, axis = 1)

plt.figure(figsize = (15, 7))
plt.plot(param_lst, train_score_mean, label = "Training scores")
plt.plot(param_lst, test_score_mean, label = "Cross Validation scores")
plt.fill_between(param_lst, train_score_mean - train_score_std,
                 train_score_mean + train_score_std, alpha=0.2)
plt.fill_between(param_lst, test_score_mean - test_score_std,
                 test_score_mean + test_score_std, alpha=0.2)
plt.legend(loc="best")
plt.title("Validation curve of Random forest classifier with varying N-estimators")
plt.xlabel("Number of estimators")
plt.ylabel("f1-score")
plt.show()


"""
Learning curve 学习曲线绘制
"""
size = np.linspace(0.1, 1, 10)
# the size of train sets
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(clf.best_params_["rforest__n_estimators"]),
                                                        X = train_X, y = train_y,
                                                        cv = cv, n_jobs = -1, train_sizes = size)
# 获取数据的学习曲线结果
train_score_mean = np.mean(train_scores, axis = 1)
train_score_std = np.std(train_scores, axis = 1)
test_score_mean = np.mean(test_scores, axis = 1)
test_score_std = np.std(test_scores, axis = 1)


plt.figure(figsize = (15, 7))
plt.plot(train_sizes, train_score_mean, label = "Training scores")
plt.plot(train_sizes, test_score_mean, label = "Cross Validation scores")
plt.fill_between(train_sizes, train_score_mean - train_score_std,
                 train_score_mean + train_score_std, alpha=0.2)
plt.fill_between(train_sizes, test_score_mean - test_score_std,
                 test_score_mean + test_score_std, alpha=0.2)
plt.legend(loc="best")
plt.title("Learning curve of Random forest classifier")
plt.xlabel("Training examples")
plt.ylabel("f1-score")
plt.show()



"""
Bias- variance trade off 偏差-方差均衡化
"""


n_samples = 250
# 样本数
iterations = 250
# 迭代次数
depth_values = range(1, 30, 5)
bias_final, variance_final = [], []
# 存储偏差和方差值的列表
for depth in depth_values:
    # 迭代每个深度值
    bias_var_df = pd.DataFrame()
    for i in range(iterations):
        # 迭代
        random_index = np.random.choice(train_X.shape[0], size=n_samples)
        X_sample = train_X[random_index]
        y_sample = train_y[random_index]
        # 生成随机数据
        classifier = RandomForestClassifier(n_estimators=clf.best_params_["rforest__n_estimators"],
                                            max_depth=depth)
        # 开发决策树分类器
        classifier.fit(X_sample, y_sample)
        # 将数据拟合到模型
        pred = classifier.predict(test_X)
        # 得到预测值
        # accuracy_score = metrics.accuracy_score(test_y, pred)
        # print("Accuracy score: {0}".format(round(accuracy_score, 3)))
        bias_var_df[str(i + 1)] = pred
        # 将预测添加到dataframe

    average_pred = bias_var_df.mean(axis=1)
    average_pred = ((test_y - average_pred) ** 2).mean()
    bias_final.append(average_pred)
    # 获得给定深度的平均预测
    average_var = bias_var_df.var(axis=1).mean()
    variance_final.append(average_var)
    # 给定深度的平均方差

plt.figure(figsize=(15, 7))
plt.plot(depth_values, bias_final, label="Bias^2")
plt.fill_between(depth_values, bias_final - np.std(bias_final),
                 bias_final + np.std(bias_final), alpha=0.3, color="b")
plt.plot(depth_values, variance_final, label="Variance")
plt.fill_between(depth_values, variance_final - np.std(variance_final),
                 variance_final + np.std(variance_final), alpha=0.3, color="g")
plt.legend(loc="best")
plt.xlabel("Tree depth")
plt.ylabel("Bias-Variance")
plt.title("Bias-Variance trade off Vs Depth (Random forest classifier)")
plt.show()


train_lenghts = train["Length"].values.reshape(len(train["Length"]), 1)
test_lengths = test["Length"].values.reshape(len(test["Length"]), 1)
train_X = np.hstack((train_X.toarray(), train_lenghts))
test_X = np.hstack((test_X.toarray(), test_lengths))
# 添加文档的长度作为特征
best_clf = RandomForestClassifier(n_estimators = clf.best_params_["rforest__n_estimators"],
                                 max_depth = 5)
# 最终参数选择最优随机森林模型
best_clf.fit(train_X, train_y)
# 将数据拟合到分类器

predict = best_clf.predict(test_X)

accuracy_score = metrics.accuracy_score(test_y, predict)
print("Accuracy score: {0}".format(round(accuracy_score, 3)))

f1_score = metrics.f1_score(test_y, predict, average = "weighted")
print("F1-score: {0}".format(round(f1_score, 3)))

print(metrics.classification_report(test_y, predict))

"""
数据降维部分，效果并不理想
"""
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
train_X_pca = pca.fit_transform(train_X)
test_X_pca = pca.transform(test_X)

clf_pca = RandomForestClassifier(n_estimators = clf.best_params_["rforest__n_estimators"], max_depth = 5)
clf_pca.fit(train_X_pca, train_y)

pred=clf_pca.predict(test_X_pca)

print(metrics.classification_report(test_y, pred))