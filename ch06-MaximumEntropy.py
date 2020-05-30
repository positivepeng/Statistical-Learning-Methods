#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/1/31 17:23
@author: phil
"""

# 参考链接：
# http://blog.csdn.net/itplus/article/details/26550369 GIS的直观解释
# https://vimsky.com/article/776.html 基于GIS的最大熵模型具体实现
from dataloader import load_01, load_all_num
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import time
from math import exp, log
import numpy as np


def preprocess_data(train_size=20000, test_size=2000):
    # 对数据进行预处理，将数据的每一维的value变为index_value的字符串，这样才能处理不同维度有相同取值的情况
    # 例如 xi = [0, 0, 123, 42]
    # 变为 xi = ["0_0", "1_0", "2_123", "3_42"]
    # 由于字符串拼接过于耗时，且考虑到本实验数据取值范围固定
    # 将xi[i]改为 [index*256+xi[0], index*256+xi[1]....]

    X_train, y_train, X_test, y_test = load_all_num(train_size=train_size, test_size=test_size)

    X_train[X_train < 128] = 0
    X_train[X_train >= 128] = 1
    X_test[X_test < 128] = 0
    X_test[X_test >= 128] = 1

    X_train_feature = np.tile(np.arange(len(X_train[0])), (len(X_train), 1)) * 256 + X_train
    X_test_feature = np.tile(np.arange(len(X_test[0])), (len(X_test), 1)) * 256 + X_test

    return X_train_feature, y_train, X_test_feature, y_test


def concat_feature_and_label(x, y):
    return int(x * 10 + y)


class MaximumEntropy:
    # 最大熵模型，参数更新使用GIS
    def __init__(self):
        self.feature2index = None  # 特征对应的下标，用于得到对应的权重值
        self.feature_count = defaultdict(int)  # 特征在训练集中出现的次数
        self.num_of_feature = None  # 特征的数量
        self.N = None  # 训练样本的数量
        self.y_values = None  # y可能的取值，用于计算P(y|X)
        self.num_of_y = None  # y取值的种类的个数
        self.w = None  # 特征函数对应的权重，每一个特征对应一个特征函数
        self.least_update = 0.001  # 最少的更新量，如果某次迭代w中每个参数更新量都少于它则终止
        self.X = None
        self.y = None
        self.data_result = None  # 特征函数关于经验分布的期望
        self.C = None   # 样本特征数量的最大值

    def init_parameter(self, X, y):
        self.X, self.y = X, y
        self.feature2index, self.feature_count = self.count_feature()
        self.num_of_feature = len(self.feature2index)
        self.N = len(X)
        self.y_values = np.unique(y)
        self.w = [0] * self.num_of_feature  # 各个特征函数对应的权重，也就是特征对应的权重
        self.data_result = self.E_data()  # f(x, y)在整个数据集上的期望只需要算一次
        self.C = max([len(Xi) for Xi in X])  # 特征数量的最大值

    def count_feature(self):
        # 统计X，y中的feature
        # 具体来说就是对于每一个数据条目 Xi, yi
        # 对于Xi中的每一个属性Xij, (Xij, yi)为一个feature
        # 这个feature对应的特征函数可理解为如果样本的第j维数据为Xij且label为yi则取值为1，否则为0
        feature_count = defaultdict(int)  # feature计数
        feature2index = {}  # 将feature对应到index的字典，用于根据feature得到对应的权重值

        for i in tqdm(range(len(self.X))):
            Xi = self.X[i]
            yi = self.y[i]
            for j in range(len(Xi)):
                # 对于一个数据条目Xi，它的每一个维度的取值Xij和yi拼接成为一个feature
                feature_count[concat_feature_and_label(Xi[j], yi)] += 1
        for index, key in enumerate(list(feature_count.keys())):
            feature2index[key] = index

        # 特征的计数验证
        assert sum(feature_count.values()) == len(self.X) * len(self.X[0])

        # print("特征数量", len(feature_count))
        # print("特征示例", list(feature_count.keys())[:10])
        return feature2index, feature_count

    def fit(self, X, y, max_iter=100):
        self.init_parameter(X, y)
        for _ in tqdm(range(max_iter)):
            new_w = self.w[:]
            model_result = self.E_model()  # 模型期望

            # 更新每个特征的权值
            for i in range(self.num_of_feature):
                new_w[i] += 1.0 / self.C * log(self.data_result[i] / (model_result[i] + 0.000000001))

            # 检查是否收敛
            update_too_little = True
            for i in range(self.num_of_feature):
                if abs(new_w[i] - self.w[i]) > self.least_update:
                    update_too_little = False
                    break
            if update_too_little:
                # 结束迭代
                break
            # 更新权重
            self.w = new_w[:]

    def E_data(self):
        # 计算特征函数(也就是特征)关于经验分布P_(x,y)的期望值
        # 即计算\sum_(x,y) P_(x,y)f(x,y)  这里P后面的下划线应该在P上面，表示P的经验分布
        # 注意上面的求和算术只是一个f对应的期望值
        # 这里的
        result = [0] * self.num_of_feature  # 每一个特征都有一个期望值
        for feature, count in self.feature_count.items():
            result[self.feature2index[feature]] += self.feature_count[feature] / self.N
        return result

    def Pyx(self, Xi):
        # p(y|x) = [\sum_(x,y) exp(w*f(x,y)) ] / Zw , Zw为归一化参数，可以先计算出分子然后再归一化
        pyx = []
        pyx_sum = 0.0
        for yi in self.y_values:
            temp_sum = 0
            # p(y|x)公式里面的求和是对所有可能的x,y的取值求和
            for xi in Xi:
                # 存在一个(xij, yi)即表明有一个特征函数取1，则应该加上对应的权重
                feature = concat_feature_and_label(xi, yi)
                if feature in self.feature2index:
                    temp_sum += self.w[self.feature2index[feature]]
            pyx.append([yi, exp(temp_sum)])
            pyx_sum += exp(temp_sum)
        # 将列表中的所有元素的第二个值转换为概率值
        pyx = list(map(lambda x: [x[0], x[1] / pyx_sum], pyx))
        return pyx

    def E_model(self):
        # 计算特征函数关于模型P(y|X)与经验分布P_(x)的期望值
        # \sum_(x,y)p_(x)p(y|x)f(x,y)
        result = [0] * self.num_of_feature  # 每个feature有一个取值

        # 上面的式子是对所有可能的x,y求和，先循环遍历所有X
        for i in range(self.N):
            Xi = self.X[i]
            # 先计算P(y|x)
            pyx = self.Pyx(Xi)
            # 再计算 \sum_(x,y)p_(x)p(y|x)f(x,y)
            for y, prob in pyx:
                for xi in Xi:
                    feature = concat_feature_and_label(xi, y)
                    if feature in self.feature_count.keys():
                        result[self.feature2index[feature]] += prob * 1.0 / self.N
        return result

    def predict(self, X):
        preds = []
        for Xi in X:
            pyx = self.Pyx(Xi)
            max_prob = 0
            pred_y = None
            for y, prob in pyx:
                if prob > max_prob:
                    pred_y = y
                    max_prob = prob
            preds.append(pred_y)
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return np.sum(preds == y.reshape(preds.shape)) / len(y)


if __name__ == "__main__":
    debug = 0

    if debug == 1:
        X = []
        y = []
        with open("ch06-data.txt", "r") as f:
            lines = f.readlines()
        for line in lines:
            splited = line.strip().split("\t")
            temp = []
            for i, feature in enumerate(splited[1:]):
                temp.append(str(i) + "_" + str(feature))
            X.append(temp)
            y.append(splited[0])
        model = MaximumEntropy()
        model.fit(X, y)
        preds = model.predict(X)
        flag = 1
        for pd, gold in zip(preds, y):
            if pd != gold:
                print("wrong", pd, gold)
                flag = 0
        if flag == 1:
            print("all right")
    else:
        X_train, y_train, X_test, y_test = preprocess_data()
        print("数据预处理完成")
        start = time.time()
        model = MaximumEntropy()
        model.fit(X_train, y_train, max_iter=100)
        print("验证集精度 %f" % model.score(X_test, y_test))
        end = time.time()

        print("过程耗时 %.2fs" % (end - start))

        """
        数据预处理完成
        100%|██████████| 1000/1000 [00:03<00:00, 295.16it/s]
        100%|██████████| 100/100 [40:38<00:00, 24.79s/it]
        验证集精度 0.860000
        过程耗时 2443.18s
        """

        """
        100%|██████████| 20000/20000 [01:16<00:00, 262.80it/s]
        100%|██████████| 100/100 [14:39:40<00:00, 604.89s/it]
        验证集精度 0.842000
        过程耗时 52886.89s
        """