#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/2/15 15:50
@author: phil
"""
import numpy as np
from dataloader import load_01
import time
from tqdm import tqdm


def default_kernel_function(x, z):
    # 默认的核函数，即计算两个向量的内积
    x = x.flatten()
    z = z.flatten()
    return np.sum(x * z)


def gaussian_kernel_function(x, z):
    # 默认情况下sigma为10
    sigma = 10
    x = x.flatten()
    z = z.flatten()
    return np.exp(-np.sum((x - z) ** 2) / (2 * sigma ** 2))


class SVM:
    """ 支持向量机 """

    def __init__(self):
        self.alpha = None  # 拉格朗日乘子
        self.b = 0  # 参数：偏置b
        self.kerner_func = None  # 核函数
        self.N = None  # 样本数量
        self.X = None
        self.y = None
        self.C = None  # 松弛变量的惩罚因子
        self.K = None  # K[i][j]表示x_i,x_j在核函数下的内积
        self.E = None  # 使用数组记录Ei值
        self.eps = 0.00001  # 精度，如果误差值在精度内则可认为两者相等

    def gx(self, x):
        # 式子7.104定义的g(x)
        result = 0.0
        for i in range(self.N):
            result += self.alpha[i] * self.y[i] * self.kerner_func(self.X[i], x)
        result += self.b
        return result

    def Ex(self, i):
        # 式子7.105定义的E(i)
        return self.gx(self.X[i]) - self.y[i]

    def choose_alpha1(self):
        # 选择第一个不满足KKT条件的拉格朗日乘子
        for i in range(self.N):
            alphai = self.alpha[i]
            gxi = self.gx(self.X[i])
            yi = self.y[i]
            # 注意这里判断KKT条件时不能使用等号判断，因为浮点运算容易产生误差
            # 也就是书中所说的 KKT条件的检验是在eps的范围内进行的
            # 也就是说如果A瞒住abs(A-B) < eps则可认为A==B
            if abs(alphai) < self.eps and yi * gxi < 1:
                # 违反KKT条件 alpha_i == 0 and y_i * g(x_i) >= 1
                return i
            elif abs(alphai - self.C) < self.eps and yi * gxi > 1:
                # 违反KKT条件 alpha_i == C and y_i*g(x_i) <= 1
                return i
            elif 0 < alphai < self.C and abs(yi * gxi - 1) > self.eps:
                # 违法KKT条件 0 < alpha_i < C and y_i*g(x_i) == 1
                return i

        # 如果没有alpha违反KKT条件则返回None
        return None

    def choose_alpha2(self, E1):
        # 选择alpha2使得|E1-E2|最大
        alpha2 = -1
        maxE2 = -1
        maxE1_E2 = -1
        for i in range(len(self.X)):
            E2 = self.E[i]
            if maxE1_E2 < abs(E1 - E2):
                maxE1_E2 = abs(E1 - E2)
                alpha2 = i
                maxE2 = E2
        return alpha2, maxE2

    def fill_K(self):
        # 填充K矩阵，K[i][j]表示kernel_function(X[i], X[j])
        self.K = np.zeros((self.N, self.N))
        for i in range(self.N):
            Xi = self.X[i]
            for j in range(i, self.N):
                self.K[i][j] = self.K[j][i] = self.kerner_func(Xi, self.X[j])

    def fill_E(self):
        # 填充E列表，E[i]表示式子7.105定义的Ei
        self.E = np.zeros((self.N, 1))
        for i in range(self.N):
            self.E[i] = self.Ex(i)

    def fit(self, X, y, C, max_iter=10, kernel_func=default_kernel_function):
        # 默认核函数是线性核函数，也就是范围两个向量的内积
        self.X, self.y, self.C = X, y, C
        self.kerner_func = kernel_func
        self.N = X.shape[0]

        # 初始化拉格朗日乘子
        self.alpha = np.zeros((self.N, 1))

        # 预先计算K矩阵
        self.fill_K()

        # 预先计算E列表
        self.fill_E()

        for _ in tqdm(range(max_iter)):
            # 首先选出不满足KKT条件的拉格朗日乘子
            alpha1 = self.choose_alpha1()
            if alpha1 is None:
                # 表示所有变量都满足KKT条件
                break
            E1 = self.E[alpha1]
            # 根据选出了的alpha1选择alpha2
            alpha2, E2 = self.choose_alpha2(E1)

            # 记录当前选出的alpha1和alpha2对应的拉格朗日乘子的值
            alpha1_old = self.alpha[alpha1]
            alpha2_old = self.alpha[alpha2]

            # 计算alpha2对应的边界
            if self.y[alpha1] != self.y[alpha2]:
                L = max(0, alpha2_old - alpha1_old)
                H = min(C, C + alpha2_old - alpha1_old)
            else:
                L = max(0, alpha2_old + alpha1_old - C)
                H = min(C, alpha2_old + alpha1_old)

            # 计算没有边界限制的情况下alpha2的取值，也就是二次函数函数值的最低点对应的自变量的取值
            eta = self.K[alpha1][alpha1] + self.K[alpha2][alpha2] - 2 * self.K[alpha1][alpha2]
            alpha2_new_uncut = alpha2_old + self.y[alpha2] * (E1 - E2) / eta

            # 根据最低点计算更新后的alpha2
            if alpha2_new_uncut > H:
                alpha2_new = H
            elif L <= alpha2_new_uncut <= H:
                alpha2_new = alpha2_new_uncut
            else:
                alpha2_new = L
            # 根据更新后的alpha2计算出更新后的alpha1
            alpha1_new = alpha1_old + self.y[alpha1] * self.y[alpha2] * (alpha2_old - alpha2_new)

            # 计算b的更新值
            b1_new = -E1 - self.y[alpha1] * self.K[alpha1][alpha1] * (alpha1_new - alpha1_old) \
                     - self.y[alpha2] * self.K[alpha2][alpha1] * (alpha2_new - alpha2_old) + self.b
            b2_new = -E2 - self.y[alpha1] * self.K[alpha1][alpha2] * (alpha1_new - alpha1_old) \
                     - self.y[alpha2] * self.K[alpha2][alpha2] * (alpha2_new - alpha2_old) + self.b

            if 0 < alpha1_new < C:
                self.b = b1_new
            elif 0 < alpha2_new < C:
                self.b = b2_new
            else:
                self.b = (b1_new + b2_new) / 2

            # 更新拉格朗日乘子
            self.alpha[alpha1] = alpha1_new
            self.alpha[alpha2] = alpha2_new

            # 更新列表E
            for i in range(self.N):
                self.E[i] = self.Ex(i)

            # 只有在使用默认核函数的情况下可以使用这种方式计算出w
            # if self.kerner_func == default_kernel_function:
            #     W = np.zeros_like(self.X[0].T)*1.0
            #     for j in range(self.N):
            #         W += self.alpha[j]*self.y[j]*self.X[j].T
            #     print("W", W)
            #     print("b", self.b)

    def predict(self, X):
        preds = []
        for Xi in X:
            gxi = self.gx(Xi)
            if gxi >= 0:
                preds.append(1)
            else:
                preds.append(-1)
        return np.array(preds)

    def score(self, X, y):
        preds = self.predict(X)
        return np.sum(preds == y.reshape(preds.shape)) / len(y)


if __name__ == "__main__":
    debug = 0
    if debug == 1:
        # 使用书上的简单例子做测试
        X = np.array([[3, 3], [4, 3], [1, 1]])
        y = np.array([[1], [1], [-1]])
        model = SVM()
        model.fit(X, y, 999999)  # 将C设置为一个比较大的数
        print("predict", model.predict(X))
    else:
        # 二分类问题，训练模型区分数据集中的0和1
        X_train, y_train, X_test, y_test = load_01(train_size=5000, test_size=1000)

        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1

        print("训练集规模", X_train.shape)
        print("测试集规模", X_test.shape)

        start = time.time()
        model = SVM()
        model.fit(X_train, y_train, C=9999, max_iter=10, kernel_func=default_kernel_function)
        end = time.time()

        print("训练集精度 %f" % model.score(X_train, y_train))
        print("验证集精度 %f" % model.score(X_test, y_test))
        print("过程耗时 %.2f" % (end - start))

        """
        训练集规模 (1042, 784)
        测试集规模 (211, 784)
        训练集精度 0.997121
        验证集精度 0.995261
        过程耗时 192.33
        """
