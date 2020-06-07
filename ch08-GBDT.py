#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/2/29 10:53
@author: phil
"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class BaseDecisionTreeRegressor:
    def __init__(self):
        self.X = None
        self.y = None
        self.best_split_col_index = None    # 最佳的切分属性的索引
        self.best_split_point = None        # 最佳的切分点的值
        self.less_than_split_point_pred = None  # 小于最佳切分点的预测值
        self.more_than_split_point_pred = None  # 大于最佳切分点的预测值

    def col_min_loss(self, col_index):
        # 求X中第i列的最佳分割点，以及其对应的loss
        # X_col  (n, 1),  y (n, 1)
        X_col = X[:, col_index]
        # 为了得到分割点，对值进行排序
        X_col_unique = np.unique(X_col)
        best_split = None
        min_loss = float("inf")
        left_pred, right_pred = 0, 0

        for i in range(len(X_col_unique)-1):
            splited_point = (X_col_unique[i] + X_col_unique[i+1]) / 2
            left_part = self.y[X_col < splited_point]
            right_part = self.y[X_col > splited_point]
            left_mean = np.mean(left_part)
            right_mean = np.mean(right_part)
            loss = np.sum((left_part-left_mean) ** 2) + np.sum((right_part-right_mean) ** 2)
            # print(splited_point, loss)
            if loss < min_loss:
                min_loss = loss
                best_split = splited_point
                left_pred = left_mean
                right_pred = right_mean
        return min_loss, best_split, left_pred, right_pred

    def fit(self, X, y):
        # X (n, m) 每一行一个样本
        self.X = X
        self.y = y
        n, m = X.shape
        best_split_col_index, best_split_point, best_loss = None, None, float("inf")
        for i in range(m):
            col_loss, col_split, left_pred, right_pred = self.col_min_loss(i)
            if col_loss < best_loss:
                best_loss = col_loss
                self.best_split_col_index = i
                self.best_split_point = col_split
                self.less_than_split_point_pred = left_pred
                self.more_than_split_point_pred = right_pred
        return best_loss

    def predict(self, X):
        # 每一行表示一条样本
        pred = []
        X_splited_col = X[:, self.best_split_col_index]
        for value in X_splited_col:
            if value > self.best_split_point:
                pred.append(self.more_than_split_point_pred)
            else:
                pred.append(self.less_than_split_point_pred)
        return np.array(pred)


class BoostingDecisionTree:
    def __init__(self):
        self.decision_trees = []

    def fit(self, X, y, max_tree_mum=6, min_loss=0.01):
        best_loss = float("inf")
        target_y = y
        for i in range(max_tree_mum):
            tree = BaseDecisionTreeRegressor()
            best_loss = tree.fit(X, target_y)
            self.decision_trees.append(tree)
            pred = self.predict(X)
            target_y = y - pred.reshape(y.shape)
            if best_loss < min_loss:
                break
        return best_loss

    def predict(self, X):
        pred = np.zeros((X.shape[0], 1))
        for tree in self.decision_trees:
            tree_pred = tree.predict(X)
            pred += tree_pred.reshape(X.shape[0], 1)
        return pred


if __name__ == "__main__":
    debug = 0
    if debug == 1:
        X = np.arange(1, 11).reshape(10, 1)
        y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]).reshape(10, 1)
        print(X.shape, y.shape)
        # 基本决策树模型
        # model = BaseDecisionTreeRegressor()
        # model.fit(X, y)
        # pred = model.predict(X)
        # print(pred)

        # boosting模型
        model = BoostingDecisionTree()
        loss = model.fit(X, y, max_tree_mum=6)
    else:
        boston = load_boston()
        X, y = boston.data, boston.target
        # std = StandardScaler()
        # y = std.fit_transform(y.reshape(-1, 1))

        model = BoostingDecisionTree()
        model.fit(X, y, max_tree_mum=100)
        print("mean square error", mean_squared_error(y, model.predict(X)))