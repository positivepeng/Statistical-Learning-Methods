#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/4/2 9:04
@author: phil
"""

import numpy as np
from scipy import stats

np.random.seed(123)


class GMM:
    def __init__(self):
        self.alpha, self.mu, self.cov = None, None, None
        self.X = None
        self.K = None

    def cal_gamma(self):
        N = len(self.X)
        gamma = np.zeros((N, self.K))  # gamma[i][j]表示第i个分布来自第j个高斯分布的概率

        for k in range(self.K):
            norm = stats.norm(self.mu[k], self.cov[k])  # 这里的cov传入的是标准差
            gamma[:, k] = self.alpha[k] * norm.pdf(self.X)
        gamma = gamma / np.sum(gamma, axis=1).reshape(N, 1)
        return gamma

    def fit(self, X, K, max_iter=2):
        # 参数K表示高斯分布的个数
        self.X = X
        self.K = K
        N = len(X)  # N个样本，每个样本D维

        # 初始化模型参数
        self.alpha = np.ones(K) / K
        self.mu = np.arange(self.K) * 1.0
        self.cov = np.ones(K)

        for _ in range(max_iter):
            # E-Step
            gamma = self.cal_gamma()

            # M-Step
            new_alpha = np.sum(gamma, axis=0) / N
            new_mu = np.zeros_like(self.mu)
            new_conv = np.zeros_like(self.cov)

            for k in range(K):
                gamma_k_sum = np.sum(gamma[:, k])
                new_mu[k] = np.sum(gamma[:, k] * self.X) / gamma_k_sum
                new_conv[k] = np.sqrt(np.sum(gamma[:, k] * (self.X - self.mu[k]) ** 2) / gamma_k_sum)

            self.alpha, self.mu, self.cov = new_alpha, new_mu, new_conv

        return self.alpha, self.mu, self.cov


if __name__ == "__main__":
    K = 3
    alpha1, mu1, std1 = 0.3, -2, 0.5
    alpha2, mu2, std2 = 0.6, 0.5, 1
    alpha3, mu3, std3 = 0.1, 4, 9

    N = 1000  # 总数据点个数
    # np.random.normal中loc含义为均值，scale为标准差
    sample1 = np.random.normal(loc=mu1, scale=std1, size=int(N * alpha1))
    sample2 = np.random.normal(loc=mu2, scale=std2, size=int(N * alpha2))
    sample3 = np.random.normal(loc=mu3, scale=std3, size=int(N * alpha3))
    X = np.concatenate((sample1, sample2, sample3), axis=0)

    np.random.shuffle(X)  # 打乱数据顺序

    print("alpha", alpha1, alpha2, alpha3)
    print("mu", mu1, mu2, mu3)
    print("cov", std1, std2, std3)
    model = GMM()
    alpha_pred, mu_pred, cov_pred = model.fit(X, K=K, max_iter=200)
    print("pred alpha:", alpha_pred)
    print("pred_mu:", mu_pred)
    print("pred_cov:", cov_pred)
