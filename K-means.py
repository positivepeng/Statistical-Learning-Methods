#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/6/7 15:38
@author: phil
"""

from sklearn.datasets import load_iris
import numpy as np


def cal_dist(v1, v2):
    return np.sum((v1 - v2) ** 2)


class Kmeans:
    def __init__(self, K):
        self.K = K
        self.kmeans = None

    def fit(self, X, max_iter=100):
        n, m = X.shape

        # 初始化k个均值向量
        # 随机选k个样本做中心
        k_indexs = np.random.randint(0, n, self.K)
        self.kmeans = X[k_indexs]

        # 不断迭代直到类别不再改变或者达到最大迭代次数
        cluster = np.array([0] * n)
        for _ in range(max_iter):
            has_change = 0
            for i in range(n):
                nearest, dist = None, float("inf")
                for j in range(self.K):
                    temp_dist = cal_dist(self.kmeans[j], X[i])
                    if temp_dist < dist:
                        dist = temp_dist
                        nearest = j
                if nearest != cluster[i]:
                    has_change = 1
                    cluster[i] = nearest

            if has_change == 0:  # 本轮迭代没有更新
                break
            # 计算新的means
            for i in range(self.K):
                if np.sum(cluster == i) > 0:
                    self.kmeans[i] = np.mean(X[cluster == i], axis=0)

    def predict(self, X):
        preds = []
        for Xi in X:
            nearest, dist = None, float("inf")
            for k in range(self.K):
                temp_dist = cal_dist(self.kmeans[k], Xi)
                if temp_dist < dist:
                    dist = temp_dist
                    nearest = k
            preds.append(nearest)
        return np.array(preds)


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target

    model = Kmeans(K=3)
    model.fit(X)
    preds = model.predict(X)
    print("聚类结果", preds)
    print("真实标签", y)