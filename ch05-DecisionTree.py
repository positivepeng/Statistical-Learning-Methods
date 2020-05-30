#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/1/23 14:47
@author: phil
"""
import numpy as np
from dataloader import load_all_num
import time


class Node:
    def __init__(self, split_index, pred_label=None):
        self.split_index = split_index  # 如果是中间节点则为用于分割的属性的索引，否则为None
        self.pred_label = pred_label  # 如果是叶节点则对应预测label，否则为None
        self.children = {}  # 存储孩子节点信息的字典，key为用于分割的属性的取值，value为子树


def empirical_entropy(col):
    # 计算col列数据对应的经验熵
    unique = np.unique(col)  # col列所有可能的取值
    ans = 0.0  # 经验熵
    for i in range(len(unique)):
        p = np.sum(col == unique[i]) / len(col)
        ans += -p * np.log2(p)
    return ans


def empirical_conditional_entropy(feature_col, label_col):
    # 计算经验条件熵，feature_col为属性取值，label_col为标签列取值
    unique = np.unique(feature_col)
    ans = 0.0
    for i in range(len(unique)):
        feature_p = np.sum(unique[i] == feature_col) / len(feature_col)
        ans += feature_p * empirical_entropy(label_col[np.where(feature_col == unique[i])])
    return empirical_entropy(label_col) - ans


class DecisionTree:
    def __init__(self):
        self.root = None

    def fit(self, X, y, tree_type="ID3"):
        # 使用X，y构建决策树，tree_type指定树的类型
        def build_tree(X, y, used_index, tree_type="ID3"):
            # 当所有的数据点label都一样的时候则产生叶节点
            if np.all(y == y[0]):
                return Node(split_index=None, pred_label=y[0])
            # 当所有的feature都已用于分割子树时则产生叶节点，预测label为当前节点对应的数据中出现次数最多的label
            if len(used_index) == len(X[0]):
                unique, counts = np.unique(y, return_counts=True)
                return Node(split_index=None, pred_label=unique[np.argmax(counts)])
            if tree_type == "ID3":
                # ID3根据信息增益选择当前样本集的切分属性
                feature_gain_ratio = []
                # 计算每个属性的信息增益
                for feature_index in range(len(X[0])):
                    feature_gain_ratio.append(empirical_conditional_entropy(X[:, feature_index], y))
                # 在没有使用过属性中选择信息增益最大的属性作为切分属性
                feature_gain_ratio = np.array(feature_gain_ratio)
                if len(used_index) > 0:
                    feature_gain_ratio[np.array(used_index)] = 0  # 先将所有已经使用过的属性的信息增益设置为0
                split_index = feature_gain_ratio.argmax()  # 选出信息增益最大的属性作为切分属性，这里没有设置信息增益阈值
                # 将已用于切分的属性加入到列表中
                used_index.append(split_index)
                # 根据选择的属性进行分割
                node = Node(split_index=split_index, pred_label=None)
                unique = np.unique(X[:, split_index])
                for i in range(len(unique)):
                    selected_index = np.where(X[:, split_index] == unique[i])
                    node.children[unique[i]] = build_tree(X[selected_index], y[selected_index], used_index[:],
                                                          tree_type=tree_type)
                return node
            elif tree_type == "C4.5":
                # C4.5根据信息增益比选择当前样本集的切分属性
                pass
            elif tree_type == "CART":
                # CART当属性为离散值时选择基尼指数选择当前样本集的切分属性
                # 当属性为连续值时使用平方误差最小化准则选择切分属性
                pass

        self.root = build_tree(X, y, [], tree_type=tree_type)

    def predict(self, X):
        # 根据决策树对X进行预测
        def predict_helper(Xi, node):
            if node.pred_label is not None:
                return node.pred_label
            split_value = Xi[node.split_index]
            return predict_helper(Xi, node.children[split_value])

        preds = []
        for i in range(X.shape[0]):
            preds.append(predict_helper(X[i, :], self.root))
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return np.sum(preds == y.reshape(preds.shape)) / len(preds)


if __name__ == "__main__":
    debug = 0

    if debug == 1:
        path = r"ch05-data.txt"
        X = []
        y = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                splited = line.strip().split(" ")
                X.append(splited[:-1])
                y.append(splited[-1])
        X = np.array(X)
        y = np.array(y)
        print("原始数据")
        print(X)
        print(y)
        model = DecisionTree()
        model.fit(X, y)
        print("预测值:")
        print(model.predict(X))
    else:
        X_train, y_train, X_test, y_test = load_all_num(train_size=10000, test_size=2000)

        X_train[X_train < 128] = 0
        X_train[X_train >= 128] = 1
        X_test[X_test < 128] = 0
        X_test[X_test >= 128] = 1

        print("训练集规模", X_train.shape)
        print("测试集规模", X_test.shape)

        start = time.time()
        model = DecisionTree()
        model.fit(X_train, y_train)
        print("验证集精度 %.2f" % model.score(X_test, y_test))
        end = time.time()

        print("过程耗时 %.2fs" % (end - start))

        """
        训练集规模(60000, 784)
        测试集规模(10000, 784)
        验证集精度 0.87
        过程耗时 696.01s
        """
