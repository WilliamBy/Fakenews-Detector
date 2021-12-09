import pandas as pd
import numpy as np
import re
from gensim.models import word2vec
import jieba
import os
import pickle


# 过滤分词列表中的停用词
def stopwords_filter(stopwords_list, seg_list):
    filter_words_list = []
    # 停用词过滤
    for word in seg_list:
        if word not in stopwords_list:
            filter_words_list.append(word)
    return filter_words_list


# 中文段落分词，返回词语列表（包含停用词过滤）
def sentence_seg(sentence):
    pattern = re.compile("[^\u4e00-\u9fa5]+")
    # 以下两行过滤出中文及字符串以外的其他符号
    sentence = pattern.sub('', sentence)
    return stopwords_filter(pd.read_table('dataset/cn_stopwords.txt', header=None).iloc[:, :].values,
                            jieba.cut(sentence))


# 新闻csv预处理成特征向量和标签
def csv2vec(csv_path, is_train=True):
    df = pd.read_csv(csv_path)  # 读取数据

    # 数据清理
    df.drop(axis=1, inplace=True, columns=["Unnamed: 0"])  # 删除索引列
    df = df.replace(re.compile(r'\[.*?\]'), " ", regex=True)  # 去除[xxx]
    df = df.replace(re.compile(r'@.*?:'), " ", regex=True)  # 去除@xxx
    df = df.replace("\t", " ", regex=False)  # 去除转义字符
    df = df.replace("网页链接", " ", regex=False)  # 去除网页链接
    df['content'] = df['content'].str.strip()  # 去除首尾空格
    df = df.fillna(value=' ')   # 填充空值

    # 内容分词和评论分词
    df['content'] = df['content'].apply(lambda x: ' '.join(sentence_seg(x)))
    df['comment_all'] = df['comment_all'].apply(lambda x: ' '.join(sentence_seg(x)))
    df = df.fillna(value=' ')  # 填充空值

    # 根据新闻语料构建词向量模型
    content_seglist = [x.split(' ') for x in df['content']]
    comment_seglist = [x.split(' ') for x in df['comment_all']]
    wv_model = None
    wv_size = 50
    if is_train:
        wv_model = word2vec.Word2Vec(content_seglist + comment_seglist, vector_size=wv_size, min_count=1)
        with open('model/wv.model', 'wb') as outfile:
            pickle.dump(wv_model, outfile)  # 保存词向量
    else:
        with open('model/wv.model', 'rb') as infile:
            wv_model = pickle.load(infile)  # 载入词向量

    # 提取新闻特征向量
    feature = []
    for i in range(len(content_seglist)):
        feature_vec = np.zeros(shape=[0], dtype='float32')  # 2n维特征向量
        text_vec = np.zeros(shape=[wv_size], dtype='float32')  # 文本向量(n维)，采用n维词向量的平均值
        count = 0  # 词数量
        for word in content_seglist[i]:
            if wv_model.wv.has_index_for(word):
                text_vec += wv_model.wv[word]   # 词向量累加
                count += 1
        if count != 0:
            feature_vec = np.concatenate((feature_vec, text_vec / count))
        else:
            feature_vec = np.concatenate((feature_vec, text_vec))

        text_vec = np.zeros(shape=[wv_size], dtype='float32')  # 文本向量(n维)，采用n维词向量的平均值
        count = 0  # 词数量
        for word in comment_seglist[i]:
            if wv_model.wv.has_index_for(word):
                text_vec += wv_model.wv[word]   # 词向量累加
                count += 1
        if count != 0:
            feature_vec = np.concatenate((feature_vec, text_vec / count))
        else:
            feature_vec = np.concatenate((feature_vec, text_vec))
        feature.append(feature_vec.tolist())

    label = []
    if is_train:    # 对于训练集还要返回label集合
        for x in df['label']:
            label.append(x)
    return {'X': np.array(feature), 'y': np.array(label)}


# 预处理
train_set = csv2vec('dataset/train.csv', is_train=True)
test_set = csv2vec('dataset/test.csv', is_train=False)
# 保存结果
with open('dataset/train.pkl', 'wb') as file:
    pickle.dump(train_set, file)
with open('dataset/test.pkl', 'wb') as file:
    pickle.dump(test_set, file)
