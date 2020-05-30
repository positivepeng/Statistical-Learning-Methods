#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2019/12/31 16:45
@author: phil
"""
import numpy as np
from dataloader import load_01, load_all_num
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


class Perceptron:
    """ 感知机模型，支持两种学习方式，原始求解和对偶问题求解 """
    def __init__(self, learning_rate, max_iter=100):
        # 设置学习率
        self.learning_rate = learning_rate
        # 设置最大的迭代轮数
        self.max_iter = max_iter
        # 权重和偏置先设为None
        self.weight = None
        self.bias = None
        # 用于对偶问题学习
        self.alpha = None

    def fit(self, X, y):
        # 根据数据训练感知机模型
        # X 每一行是一个样本
        n, m = X.shape  # 表示有n个样本，每个样本为一个m维向量
        self.weight = np.zeros(m)  # 初始化权重为0
        self.bias = 0  # 初始化偏置为0
        losses = []

        for _ in tqdm(range(self.max_iter)):
            # 每一轮迭代都从头开始找，直到找到分类错误的样本
            for j in range(n):
                if (np.sum(self.weight * X[j]) + self.bias) * y[j] <= 0:  # 找到分类错误的样本
                    # 使用分类错误的样本更新权重和偏置
                    self.weight += self.learning_rate * y[j] * X[j]
                    self.bias += self.learning_rate * y[j]
                    # break  # 更新完成后可以选择break，下一次又重头开始找，也可以选择不break但是这样max_iter的含义就有所不同
            loss = 0
            for j in range(n):
                fy = (np.sum(self.weight * X[j]) + self.bias) * y[j]
                if fy < 0:
                    loss -= fy
            losses.append(loss)
        return losses

    def fit_dual(self, X, y):
        # 求解感知机的对偶问题
        # Question: 为什么要提出对偶问题？
        # 一种解释为:
        #   考虑m很大，n很小的情况，也就是单条数据的维度很大，而数据量很小
        #   这种情况下使用原问题求解每次判断需要计算w*x，这样计算复杂为O(m)
        #   而使用这种办法，预先计算好Gram矩阵的情况下，判断时需计算Gram[j]*self.alpha*y，O(n)
        n, m = X.shape
        # 先计算出任意两个X向量的内积，也就是Gram矩阵
        Gram = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Gram[i][j] = np.sum(X[i] * X[j])
        # alpha矩阵初始化为全0
        # alpha[i]/self.learning_rate表示第i个样本被用于更新梯度的次数
        self.alpha = np.zeros(n)
        self.bias = 0
        for _ in tqdm(range(self.max_iter)):
            for j in range(n):
                if (np.sum(Gram[j] * self.alpha * y) + self.bias) * y[j] <= 0:  # 分类错误
                    # 样本j被用于更新梯度就加一个learning_rate
                    # 因为alpha/learning_rate表示的才是被用于梯度更新的次数
                    self.alpha[j] += self.learning_rate
                    self.bias += self.learning_rate * y[j]
                    # break
        # 通过self.alpha计算self.weight，用于预测
        # 注意这里的求和，X.T每一列表示一个样本，self.alpha和y相乘后结果的shape为(1, n)
        # self.alpha的y相乘后的结果再乘X.T可理解为对所有样本进行加权求和
        self.weight = np.sum(self.alpha * y * X.T, axis=1)

    def predict(self, X):
        pred = []
        for i in range(len(X)):
            pred.append(sign(np.sum(self.weight * X[i]) + self.bias))
        return np.array(pred)

    def score(self, X, y):
        preds = self.predict(X)
        return np.sum(preds == y.reshape(preds.shape)) / len(y)


if __name__ == "__main__":
    debug = 2

    if debug == 0:
        # 书上的案例
        X = np.array([[3, 3], [4, 3], [1, 1]])
        y = np.array([1, 1, -1])
        model = Perceptron(learning_rate=1)
        # model.fit(X, y)
        model.fit_dual(X, y)
        print("参数为:", model.weight, model.bias)
        print("预测结果为:", model.predict(X))
        """
        参数为: [1. 1.] -3
        预测结果为: [ 1  1 -1]
        """
    elif debug == 1:
        # 二分类问题，将数字分为大于等于5的和小于5的
        X_train, y_train, X_test, y_test = load_all_num()
        # *!* 这里有个坑 *!*，先改变小于5的再变大于5的
        y_train[y_train < 5] = -1
        y_train[y_train >= 5] = 1
        y_test[y_test < 5] = -1
        y_test[y_test >= 5] = 1

        print("训练集规模", X_train.shape)
        print("测试集规模", X_test.shape)

        start = time.time()
        model = Perceptron(learning_rate=0.001, max_iter=100)

        history = model.fit(X_train, y_train)
        plt.plot(np.arange(len(history)), np.array(history))
        plt.show()
        print("验证集精度 %.2f" % model.score(X_test, y_test))
        print("过程耗时 %.2f" % (time.time() - start))

        """
        训练集规模 (60000, 784)
        测试集规模 (10000, 784)
        验证集精度 0.76
        过程耗时 177.86
        """

    elif debug == 2:
        # 二分类问题，训练模型区分数据集中的0和1
        X_train, y_train, X_test, y_test = load_01()
        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1

        print("训练集规模", X_train.shape)
        print("训练集正例", np.sum(y_train == 1))
        print("训练集负例", np.sum(y_train == -1))
        print("测试集规模", X_test.shape)
        print("测试集正例", np.sum(y_test == 1))
        print("测试集负例", np.sum(y_test == -1))

        start = time.time()
        model = Perceptron(learning_rate=0.001, max_iter=20)
        history = model.fit(X_train, y_train)
        end = time.time()

        plt.plot(np.arange(len(history)), np.array(history))
        plt.show()

        print("训练集精度 %f" % model.score(X_train, y_train))
        print("验证集精度 %f" % model.score(X_test, y_test))
        print("过程耗时 %.2f" % (end - start))

        """
        训练集规模 (12665, 784)
        训练集正例 6742
        训练集负例 5923
        测试集规模 (2115, 784)
        测试集正例 1135
        测试集负例 980
        训练集精度 1.000000
        验证集精度 0.999054
        过程耗时 6.28
        """

