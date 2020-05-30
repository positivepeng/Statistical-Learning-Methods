#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/1/31 17:22
@author: phil
"""

import numpy as np
from random import shuffle
from dataloader import load_01, load_all_num
import time
import matplotlib.pyplot as plt
import math
from tqdm import tqdm


class LogisticRegression:
    def __init__(self):
        self.W = None

    def fit(self, X, y, learning_rate=0.01, epoch=10):
        # X每一行是一个样本
        n, dim = X.shape
        # 初始化参数
        self.W = np.zeros((dim, 1))
        # 保存训练过程中的loss值
        losses = []
        # 这里使用随机梯度下降算法，每次选择一个样本对参数进行更新
        for _ in tqdm(range(epoch)):
            # 打乱训练集顺序
            index = list(range(n))
            shuffle(index)
            X, y = X[index], y[index]
            for j in range(n):
                Xj = X[j].reshape(1, -1)
                yj = y[j]
                # 原始计算公式
                # e_Wx = np.exp(Xj.dot(self.W))
                # p1 = e_Wx / (1+e_Wx)
                # 避免数值溢出
                p1 = 1 / (1 + np.exp(-Xj.dot(self.W)))
                self.W += learning_rate * (yj - p1) * Xj.T
            loss = 0
            for j in range(n):
                Xj = X[j].reshape(1, -1)
                yj = y[j]
                p1 = 1 / (1 + np.exp(-Xj.dot(self.W)))
                loss -= yj * math.log2(p1) + (1 - yj) * math.log2(1 - p1)
            losses.append(loss)
        return losses

    def predict(self, X):
        preds = []
        for i in range(len(X)):
            Xi = X[i]
            # 见fit函数
            # e_Wx = np.exp(Xi.dot(self.W))
            # p1 = e_Wx / (1+e_Wx)
            p1 = 1 / (1 + np.exp(-Xi.dot(self.W)))
            if p1 >= 0.5:
                preds.append(1)
            else:
                preds.append(0)
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return np.sum(preds == y.reshape(preds.shape)) / len(y)


if __name__ == "__main__":
    debug = 1

    if debug == 1:
        # 二分类问题，训练模型区分数据集中的0和1
        X_train, y_train, X_test, y_test = load_01()
        """
            训练集规模 (12665, 785)
            测试集规模 (2115, 785)
            100%|██████████| 100/100 [00:50<00:00,  1.99it/s]
            训练集精度 0.999210
            验证集精度 0.999054
            过程耗时 50.25
        """
    else:
        # 二分类，训练模型区分数据集中的>=5和<5的数字
        X_train, y_train, X_test, y_test = load_all_num()
        y_train[y_train < 5] = 0
        y_train[y_train >= 5] = 1
        y_test[y_test < 5] = 0
        y_test[y_test >= 5] = 1
        """
            训练集规模 (60000, 785)
            测试集规模 (10000, 785)
            训练集精度 0.87
            验证集精度 0.87
            过程耗时 229.25
        """
    # 如果直接使用原始数据，每个样本中的数据值取值范围为[0,255]，计算过程会使得exp(wx)过大导致结算结果出错
    # 有三种解决办法
    # 方法1：将p(y=1|x)=exp(wx)/(1+exp(wx))改写为p(y=1|x)=1/(1+exp(-wx))，见fit函数
    # 方法2：将输入数据归一化 X = X / 255，使得属性取值范围缩到[0,1]
    # X_train = X_train / 255
    # X_test = X_test / 255
    # 方法3：将输入数据标准化，代码如下：注意测试集的标准化参数来自训练集
    # stdSca = StandardScaler()
    # X_train = stdSca.fit_transform(X_train)
    # X_test = stdSca.transform(X_test)

    train_one_column = np.ones((X_train.shape[0], 1))
    test_one_column = np.ones((X_test.shape[0], 1))

    # 这里除255是为了避免在计算loss时出现log(0)
    X_train = np.c_[X_train, train_one_column] / 255
    X_test = np.c_[X_test, test_one_column] / 255

    print("训练集规模", X_train.shape)
    print("测试集规模", X_test.shape)

    start = time.time()
    model = LogisticRegression()
    history = model.fit(X_train, y_train, learning_rate=0.001, epoch=100)
    end = time.time()

    plt.plot(np.arange(len(history)), np.array(history))
    plt.show()

    print("训练集精度 %f" % model.score(X_train, y_train))
    print("验证集精度 %f" % model.score(X_test, y_test))
    print("过程耗时 %.2f" % (end - start))
