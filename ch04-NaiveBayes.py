#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/1/21 22:37
@author: phil
"""
import numpy as np
from dataloader import load_all_num
import time
import math
from tqdm import tqdm


def change_feature_to_index(index, feature_value, y_value):
    return str(index) + "_" + str(feature_value) + "|" + str(y_value)


class NaiveBayes:
    """朴素贝叶斯模型"""
    def __init__(self):
        self.eps = 0.0000000001

    def fit(self, X, y, laplace_smoothing=False):
        n, m = X.shape   # n条数据，每条m维
        # X和y都是ndarray
        self.Xy = np.c_[X, y]
        # 所有可能的label
        self.y_unique = np.unique(self.Xy[:, -1])
        print("所有可能的标签取值包括", self.y_unique)

        self.cond_prob = {}
        for yi in self.y_unique:
            yi_cnt = np.sum(self.Xy[:, -1] == yi)
            if laplace_smoothing:  # 进行Laplace修正
                self.cond_prob[str(yi)] = (yi_cnt + 1) / (n + len(self.y_unique))
            else:
                self.cond_prob[str(yi)] = yi_cnt / n

        for i in tqdm(range(len(X[0]))):
            X_icol = X[:, i]
            X_icol_unique = np.unique(X_icol)
            for feature in X_icol_unique:
                # 遍历所有的label，确定y_j，然后可计算出P(X_i|y_j)
                for yv in self.y_unique:
                    # 选出训练集中所有label为y_unique[y_index]的数据
                    y_yv_lines = self.Xy[self.Xy[:, -1] == yv]
                    # 在选出的数据中计算第feature_index列的特征对应的取值为feature的数量
                    feature_cnt = np.sum([y_yv_lines[:, i] == feature])
                    combine_feature_index = change_feature_to_index(i, feature, yv)

                    if laplace_smoothing:
                        self.cond_prob[combine_feature_index] = (feature_cnt + 1) / (len(y_yv_lines) + len(X_icol_unique))
                    else:
                        self.cond_prob[combine_feature_index] = feature_cnt / len(y_yv_lines)

        for k in self.cond_prob.keys():
            self.cond_prob[k] = math.log2(self.cond_prob[k]+self.eps)

    def predict(self, X):
        pred = []
        for i in range(len(X)):
            pred.append(self.predict_one(X[i]))
        return np.array(pred)

    def predict_one(self, Xi):
        # 对输入的Xi进行预测，计算将Xi预测为各个label的后验概率
        # P(y|X) = P(X|y)*P(y) / p(X) 注意计算过程中并未概率分母P(X)
        # 所以下面的代码就是对各个label，计算P(X|y)*P(y)，然后比较大小，将后验概率最大的label作为预测标签
        # P(y)即为先验概率，也就是数据集中样本的标签的频率分布
        # P(X|y)根据独立性假设可分解为P(X_1|y)*P(X_2|y)...P(X_k|y)

        prob = []       # 预测为各个label的概率
        # 先计算各个label的先验概率，P(y)
        for yi in self.y_unique:
            prob.append(self.cond_prob[str(yi)])
        # print("先验概率", prob)
        # 计算P(X|y)
        for i in range(len(Xi)):
            # Xi对应feature_index维的取值，确定X_i
            feature = Xi[i]
            # 遍历所有的label，确定y_j，然后可计算出P(X_i|y_j)
            for y_index in range(len(self.y_unique)):
                combine_feature_index = change_feature_to_index(i, feature, self.y_unique[y_index])
                if combine_feature_index in self.cond_prob:
                    prob[y_index] += self.cond_prob[combine_feature_index]
                else:
                    prob[y_index] += math.log2(self.eps)
        # print("后验概率", prob)
        return self.y_unique[np.argmax(np.array(prob))] # 返回后验概率最大的label作为预测值

    def score(self, X, y):
        preds = self.predict(X)
        return np.sum(preds == y.reshape(preds.shape)) / len(y)


if __name__ == "__main__":
    debug = 0

    if debug == 1:
        X = np.array([[1, "S"], [1, "M"], [1, "M"], [1, "S"], [1, "S"],
                      [2, "S"], [2, "M"], [2, "M"], [2, "L"], [2, "L"],
                      [3, "L"], [3, "M"], [3, "M"], [3, "L"], [3, "L"],])
        y = np.array([[-1], [-1], [1], [1], [-1],
                      [-1], [-1], [1], [1], [1],
                      [1], [1], [1], [1], [-1]])
        model = NaiveBayes()
        model.fit(X, y, laplace_smoothing=True)
        print(model.cond_prob)
        y = model.predict(np.array([[2, "S"]]))
        print(y)
    else:
        X_train, y_train, X_test, y_test = load_all_num()

        X_train[X_train < 128] = 0
        X_train[X_train >= 128] = 1
        X_test[X_test < 128] = 0
        X_test[X_test >= 128] = 1

        print("训练集规模", X_train.shape)
        print("测试集规模", X_test.shape)

        start = time.time()
        model = NaiveBayes()
        model.fit(X_train, y_train)
        print("验证集精度 %.2f" % model.score(X_test, y_test))
        end = time.time()

        print("过程耗时 %.2fs" % (end - start))

        """
        训练集规模 (60000, 784)
        测试集规模 (10000, 784)
        所有可能的标签取值包括 [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]
        验证集精度 0.84
        过程耗时 688.03s
        """
