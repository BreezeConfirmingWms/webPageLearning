import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from bs4 import BeautifulSoup
from pathlib import Path
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



STOPWORDS = stopwords.words('english')

label_tags = ["course", "department", "faculty", "other", "project", "staff", "student"]

class fetch_data(object):
    """
    此类有助于从中获取数据
     将 HTML 文件转换为结构化文件。
    """

    def __init__(self, directory):
        """
        类的构造函数.

        Arguments:

        1. directory: 文件目录.
        """

        self.data_frame = pd.DataFrame(columns=["URL", "Text", "University", "Label"])
        # dataframe to load data into
        self.dir = directory
        # input files directory
        self.lemmatizer = WordNetLemmatizer()
        # lemmatizer object

    def __text_process(self, text):
        """
        此功能有助于执行文本处理
         通过删除停用词和词形还原。

        Arguments:

        1. self: 对象.
        2. text: 要处理的输入文本.

        Return:
        1. text: 处理过的文本
        """

        text = text.replace("cs", "computer science")
        # 细节处理，用cs替换computer science增强语义可读性
        text_lst = text.split(" ")
        # 获取单词列表
        text_lst = [self.lemmatizer.lemmatize(word) for word in text_lst if
                    word not in STOPWORDS and len(word) > 1]  # 无关词删除和有效词的归并与调整，可作为正则化手段
        # 文本处理
        return " ".join(text_lst)

    def __get_text(self, filename):
        """
       此功能有助于获取文本
         数据，以及给定的锚文本
         HTML 文件名。

        Arguments:

        1. self: 对象
        2. filename:数据来源的文件名
         是要提取的。
        """

        with open(filename,'r',encoding='utf8') as obj:
            # 读取数据路径
            print(filename)
            data = obj.read()

        data = re.sub('^[^<]+', "", data)
        # 从文件中删除顶部标题
        data_bs = BeautifulSoup(data)
        # beautiful soup 爬虫抓取网页数据
        text = data_bs.get_text()
        # 只保留纯文本

        table = str.maketrans('', '', punctuation)
        text = text.translate(table)
        # 从数据中删除所有标点符号
        text = text.replace("\n", " ")
        # 用空格替换新行
        text = re.sub("\d", "", text)
        # 从文本中删除所有数字
        text = re.sub("[\s]{2,}", " ", text).lower()
        # 用单个空格替换多个空格
        text = self.__text_process(text)
        # func 调用以执行进一步的文本处理
        return text

    def __get_filename(self):
        """
       此函数充当生成器
         产生具有完整路径的文件名
         这有助于进一步的处理。
        """

        index = 0
        # 为dataframe设置索引
        for path, _, file_lst in os.walk(self.dir):
            # 遍历文件的子目录
            for f in file_lst:
                # 遍历子目录中的每个文件
                self.data_frame.loc[index, "URL"] = f.strip("^")
                # 将 URL 添加到数据框
                path_lst = path.split("\\")
                # 获取列表中遍历的路径
                self.data_frame.loc[index, "University"] = path_lst[-1]
                # 将University 添加到dataframe
                self.data_frame.loc[index, "Label"] = path_lst[-2]
                # 将类别添加到dataframe
                yield index, os.path.join(path, f)
                # 产生索引和文件路径
                index += 1
                # 更新索引

    def get_data(self):
        """
        该函数遍历子目录
         到每个文件名，并获取所需的数据
         预处理后。
        """

        file_generator = self.__get_filename()
        # 生成器获取文件名
        for index, file_path in file_generator:
            # 遍历索引和文件路径
            # 可选只编辑可访问网页文本.html
            # if not file_path.endswith(".html"):
            #     continue
            text = self.__get_text(file_path)
            # 获取文件的文本和锚点内容
            self.data_frame.loc[index, "Text"] = text.encode("utf8")
            # 将文本数据添加到数据框


def Process_Data(dir_path):  #html解析器爬取网页纯文本，并返回文本字符数组和标签数组
    dir_path=Path(dir_path)
    texts=[]
    labels=[]
    for label_path in label_tags:
        for html_file in (dir_path/label_path).glob('**/*.html'):
            with open(os.path.join(html_file),'rb') as f:
                trimmed = b' '.join(line.strip() for line in f.readlines()[5:])
                soup = BeautifulSoup(trimmed, 'html.parser')
                file_text = soup.get_text()
                texts.append(file_text)
            labels.append(label_tags.index(label_path))
    return texts,labels


if __name__ == "__main__":
    crawl = fetch_data(directory="webkbs")
    crawl.get_data()
    crawl.data_frame.to_csv(path_or_buf="data.csv", index=False)