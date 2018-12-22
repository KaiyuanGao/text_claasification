# coding=utf-8
# @author: kaiyuan
# blog: https://blog.csdn.net/Kaiyuan_sjtu
import numpy as np
import json
from tensorflow.contrib import learn

import json
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict

# 分词

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = WordPunctTokenizer()

# 记录每个单词及其出现的频率
word_freq = defaultdict(int)

# 读取数据集，分词，统计频率，保存
with open('../BaseUtil/data/yelp_academic_dataset_review.json', 'rb') as f:
    for line in f:
        review = json.loads(line)
        words = word_tokenizer.tokenize(review['text'])
        for word in words:
            word_freq[word] += 1

    print('load finished')

# 保存词频表
with open('word_freq.pickle', 'wb') as g:
    pickle.dump(word_freq, g)
    print(len(word_freq))
    print('word_freq save finished')

num_classes = 5
# 将词表排序，过滤次数最少的三个
sort_words = list(sorted(word_freq.items(), key=lambda x: -x[1]))
print(sort_words[:10], sort_words[-10:])

# #构建vocablary，并将出现次数小于5的单词全部去除，视为UNKNOW
vocab = {}
i = 1
vocab['UNKNOW_TOKEN'] = 0
for word, freq in word_freq.items():
    if freq > 5:
        vocab[word] = i
        i += 1
print(i)
UNKNOWN = 0

data_x = []
data_y = []
max_sent_in_doc = 30
max_word_in_sent = 30


#将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
# 不够的补零，多余的删除，并保存到最终的数据集文件之中
with open('../BaseUtil/data/yelp_academic_dataset_review.json', 'rb') as f:
    for line in f:
        doc = []
        review = json.loads(line)
        sents = sent_tokenizer.tokenize(review['text'])
        for i, sent in enumerate(sents):
            if i < max_sent_in_doc:
                word_to_index = []
                for j, word in enumerate(word_tokenizer.tokenize(sent)):
                    if j < max_word_in_sent:
                            word_to_index.append(vocab.get(word, UNKNOWN))
                doc.append(word_to_index)

        label = int(review['stars'])
        labels = [0] * num_classes
        labels[label-1] = 1
        data_y.append(labels)
        data_x.append(doc)
    pickle.dump((data_x, data_y), open('yelp_data', 'wb'))
    print(len(data_x))

def read_dataset():
    with open('yelp_data', 'rb') as f:
        data_x, data_y = pickle.load(f)
        length = len(data_x)
        train_x, dev_x = data_x[:int(length*0.9)], data_x[int(length*0.9)+1 :]
        train_y, dev_y = data_y[:int(length*0.9)], data_y[int(length*0.9)+1 :]
        return train_x, train_y, dev_x, dev_y

