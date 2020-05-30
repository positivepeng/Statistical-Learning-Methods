#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/2/29 10:53
@author: phil
"""
import numpy as np


class BaseDecisionTreeRegressor:
    def __init__(self):
        self.X = None
        self.y = None
        self.best_split_col_index = None
        self.best_split_point = None

    def col_min_loss(self, col_index):
        # 求X中第i列的最佳分割点，以及其对应的loss
        # X_col  (n, 1),  y (n, 1)
        X_col = X[:, col_index]
        # 为了得到分割点，对值进行排序
        X_col = np.sort(X_col)
        best_split = None
        min_loss = float("inf")

        for i in range(len(X_col)-1):
            splited_point = (X_col[i] + X_col[i+1]) / 2
            left_part = self.y[X_col < splited_point]
            right_part = self.y[X_col > splited_point]
            left_mean = np.mean(left_part)
            right_mean = np.mean(right_part)
            loss = np.sum((left_part-left_mean) ** 2) + np.sum((right_part-right_mean) ** 2)
            # print(splited_point, loss)
            if loss < min_loss:
                min_loss = loss
                best_split = splited_point
        return min_loss, best_split

    def fit(self, X, y):
        # X (n, m) 每一行一个样本
        self.X = X
        self.y = y
        n, m = X.shape
        best_split_col_index, best_split_point, best_loss = None, None, float("inf")
        for i in range(m):
            col_loss, col_split = self.col_min_loss(i)
            # print(col_loss, col_split)
            if col_loss < best_loss:
                best_loss = col_loss
                self.best_split_col_index = i
                self.best_split_point = col_split
        # print("best split point ", self.best_split_point)
        return best_loss

    def predict(self, X):
        # 每一行表示一条样本
        pred = []
        X_splited_col = X[:, self.best_split_col_index]

        less_than_pred_value = np.mean(self.y[X_splited_col < self.best_split_point])
        great_than_pred_value = np.mean(self.y[X_splited_col > self.best_split_point])
        for value in X_splited_col:
            if value > self.best_split_point:
                pred.append(great_than_pred_value)
            else:
                pred.append(less_than_pred_value)
        return np.array(pred)


class BoostingDecisionTree:
    def __init__(self):
        self.decision_trees = []

    def fit(self, X, y, max_tree_mum=6, min_loss=0.01):
        best_loss = 9999999
        for i in range(max_tree_mum):
            tree = BaseDecisionTreeRegressor()
            best_loss = tree.fit(X, y)
            self.decision_trees.append(tree)
            pred = tree.predict(X)
            y = (y - pred.reshape(y.shape))
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
    loss = model.fit(X, y, max_tree_mum=10)
    print("loss", loss)
    # print("pred", model.predict(X))

