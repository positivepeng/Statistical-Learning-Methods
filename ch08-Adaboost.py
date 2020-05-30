#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2019/12/25 17:03
@author: phil
"""
import numpy as np
from dataloader import load_01
import time
from tqdm import tqdm

def chooseBestSimpleModel(X, y, w):
    # 当X为多维数据时，每一个维度都可以训练一个SimpleModel
    # 选出本轮最好的SimpleMode
    min_error = float("inf")
    feature_index = None
    best_model = None
    for idx in range(len(X[0])):
        temp_model = SimpleModel()
        error = temp_model.fit(X[:, idx], y, w)
        if error < min_error:
            min_error = error
            feature_index = idx
            best_model = temp_model
    return feature_index, best_model, min_error


class Adaboost:
    def __init__(self, max_base_model_num=10):
        # 基学习器的权重
        self.alpha = []
        # 基学习器, 每个学习器都有predict方法
        self.base_models = []
        # 基学习器数量的最大值
        self.max_base_model_num = max_base_model_num

    def fit(self, X, y, debug=0):
        # 使用数据X，y训练模型
        # 类型均为ndarray, shape为(n,)
        n = len(X)
        # 初始化权重
        w = np.ones((n, 1)) / n

        for i in range(self.max_base_model_num):    # 控制最大基模型数量
            # 新增基模型，训练，预测
            idx, bestSimpleModel, error = chooseBestSimpleModel(X, y, w)
            # print("choose ", idx, " error", error)
            pred = bestSimpleModel.predict(X[:, idx])

            # 计算基模型对应的权重
            alpha_i = 0.5 * np.log((1 - error) / error)
            # 将本次训练的积模型及模型所占的权重记录下来
            self.alpha.append(alpha_i)
            self.base_models.append((idx, bestSimpleModel))
            # 更新数据对应的权重
            new_w = np.exp(-alpha_i * y * pred) * w
            new_w = new_w / np.sum(new_w)
            w = new_w
            # 使用当前的模型做预测，决定是否退出循环
            if np.sum(self.predict(X) == y) == n:
                break
            print(i)

    def predict(self, X):
        # 使用训练好的Adaboost模型进行预测
        pred = np.zeros((X.shape[0], 1)) * 1.0  # 转换为浮点数
        # 计算模型叠加和
        for alphai, (idx, base_model) in zip(self.alpha, self.base_models):
            pred += alphai * base_model.predict(X[:, idx])
        pred[pred > 0] = 1
        pred[pred < 0] = -1
        return pred

    def score(self, X, y):
        preds = self.predict(X)

        return np.sum(preds == y.reshape(preds.shape)) / len(y)


class SimpleModel:
    def __init__(self):
        # 模型由两个变量决定，一个是分割点，一个是确定分割点哪边为正例
        # best_split_point 分割点
        # less_is_positive[bool类型] True表示小于分割点的值预测为+1
        self.best_split_point = None
        self.less_is_positive = None

    def fit(self, X, y, w):
        error = np.sum(w)  # 初始化，假设所有样本都被错分类
        # 构造一个列表，列表中间部分是X从小到大排序，左右两边是增加的数
        # 最左边的比X的最小值小1，最右边的比X的最大值大1
        # 目的是为了直接通过(X[i]+X[i+1])/2得到所有可能的分割点
        sorted_X = list(X)
        sorted_X.append(min(X) - 1)
        sorted_X.append(max(X) + 1)
        sorted_X.sort()

        # 遍历所有的分割点
        for i in range(len(sorted_X) - 1):
            split_point = (sorted_X[i] + sorted_X[i + 1]) / 2
            temp_pred = np.ones_like(w)
            # 小于为分割点预测为正的情况
            temp_pred[X < split_point] = 1
            temp_pred[X > split_point] = -1

            if np.sum((temp_pred != y) * w) < error:
                self.best_split_point = split_point
                self.less_is_positive = True
                error = np.sum((temp_pred != y) * w)
            # 大于为分割点预测为正的情况
            temp_pred[X > split_point] = 1
            temp_pred[X < split_point] = -1
            if np.sum((temp_pred != y) * w) < error:
                self.best_split_point = split_point
                self.less_is_positive = False
                error = np.sum((temp_pred != y) * w)
        return error

    def predict(self, X):
        # 根据分割点做预测，等于分割点时为预测为+1
        pred = np.ones((X.shape[0], 1))
        X = X.reshape(X.shape[0], 1)
        if self.less_is_positive:
            pred[X > self.best_split_point] = -1
        else:
            pred[X < self.best_split_point] = -1
        return pred


if __name__ == "__main__":
    debug = 0

    if debug == 1:
        X = np.arange(10).reshape(10, 1)
        y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1]).reshape(10, 1)
        model = Adaboost()
        model.fit(X, y, debug=0)
        print("训练集精度 %f" % model.score(X, y))
    else:
        # 二分类问题，训练模型区分数据集中的0和1
        X_train, y_train, X_test, y_test = load_01(train_size=5000, test_size=1000)
        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1

        X_train[X_train < 128] = 0
        X_train[X_train >= 128] = 1
        X_test[X_test < 128] = 0
        X_test[X_test >= 128] = 1

        print("训练集规模", X_train.shape)
        print("测试集规模", X_test.shape)

        start = time.time()
        model = Adaboost(max_base_model_num=10)
        model.fit(X_train, y_train)
        end = time.time()

        print("训练集精度 %f" % model.score(X_train, y_train))
        print("验证集精度 %f" % model.score(X_test, y_test))
        print("过程耗时 %.2f" % (end - start))

        """
        训练集规模(1042, 784)
        测试集规模(211, 784)
        训练集精度 1.000000
        验证集精度 0.995261
        过程耗时 830.51
        """

