from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from itertools import product
from nltk.stem import WordNetLemmatizer

import pandas as pd
from FetchData import label_tags,Process_Data



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m','--method',type=str,default="StopText",help='The chosen method of preprocess Text Data,PlainText and StopText')
opt = parser.parse_args()

method=opt.method


def add_url_to_text(data):
    data["URL"] = data["URL"].lower().replace(".html", "")

    data["URL"] = re.sub("[\d-]", "", data["URL"])

    data["URL"] = re.sub("_", ":", data["URL"])
    data["URL"] = data["URL"].replace("^", "/", 20)

    url_keys_lst = data["URL"].split("/")[3:]

    url_keys_lst = map(lambda x: lemma.lemmatize(x.replace("cs", "computer science")), url_keys_lst)

    data["Text"] = "{1} {0}".format(data["Text"], " ".join(url_keys_lst))

    return data


if method=="PlainText":
    dir_path='./webkbs'
    txts,labs=Process_Data(dir_path)
    X_train,X_test,y_train,y_test=train_test_split(txts,labs,test_size=.35,stratify=labs)

if method == "StopText":
    lemma = WordNetLemmatizer()

    df = pd.read_csv(filepath_or_buffer="data.csv")

    df.dropna(how="any", inplace=True)




    df = df.apply(func=add_url_to_text, axis=1)
    df["Label"] = LabelEncoder().fit_transform(df["Label"])

    Train_df, Test_df, y_train, y_test = train_test_split(df, df["Label"], test_size=.2, stratify=df["Label"])
    # X_train,Valid_data,y_train,Valid_tag=train_test_split(X_train,y_train,test_size=2/9,stratify=y_train)

    X_train = Train_df["Text"]
    X_test = Test_df["Text"]

print("data loaded")

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC(kernel='rbf')),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test)

accuracy_score = metrics.accuracy_score(y_test, predicted)
print("Accuracy score: {0}".format(round(accuracy_score, 3)))
print(metrics.classification_report(y_test, predicted))