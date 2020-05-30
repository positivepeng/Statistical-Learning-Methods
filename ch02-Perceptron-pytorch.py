#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2019/12/31 21:17
@author: phil
"""

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

# 设置随机种子
torch.manual_seed(42)


class PerceptronModel(nn.Module):
    # 利用pytorch的自动求导机制实现感知机模型
    def __init__(self, learning_rate, max_iter=1000):
        super(PerceptronModel, self).__init__()
        self.learning_rate = learning_rate  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        # 模型参数
        self.weight = None
        self.bias = None

    def forward(self, X, y, method="grad"):
        # 参数method指定参数更新的方法
        n, m = X.shape
        # 初始化模型参数
        self.weight = torch.zeros(m, dtype=torch.float32, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)
        # 将输入的ndarray转换为tensor
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        for i in range(self.max_iter):
            # 标记本轮迭代是否遇到错分类样本
            has_error = 0
            # 遍历训练样本
            for idx in range(n):
                Xi = X[idx, :]
                yi = y[idx]
                # 计算线性函数输出值
                out = torch.add(torch.sum(self.weight * Xi), self.bias)
                if out * yi <= 0:  # 分类错误则更新
                    has_error = 1  # 标记本轮循环遇到了错误样本
                    loss = torch.max(torch.tensor(0, dtype=torch.float32), -1 * out * yi)
                    if method == "grad":
                        # 直接求导，也就是直接算出dloss/dweight，dloss/dbiad
                        gradw = torch.autograd.grad(loss, self.weight, retain_graph=True)
                        gradb = torch.autograd.grad(loss, self.bias, retain_graph=True)
                        with torch.no_grad():
                            # 注意这里更新参数时对参数的计算无需求导
                            self.weight -= self.learning_rate * gradw[0]
                            self.bias -= self.learning_rate * gradb[0]
                    if method == "backward":
                        # 误差反向传播求导，直接调用backward计算出所有叶节点的梯度
                        loss.backward()
                        with torch.no_grad():
                            self.weight -= self.learning_rate * self.weight.grad
                            self.bias -= self.learning_rate * self.bias.grad
                            # 梯度重置为0
                            self.weight.grad.zero_()
                            self.bias.grad.zero_()
                    if method == "optimizer":
                        # 利用优化器完成参数的更新
                        # 1. 误差反向传播，也就是计算出所有计算图中各个结点的梯度
                        loss.backward()
                        # 2. 定义优化器
                        optimizer = optim.SGD((self.weight, self.bias), lr=self.learning_rate)
                        # 3. 根据上一步计算出的梯度更新
                        optimizer.step()
                        # 4. 将梯度清空
                        optimizer.zero_grad()
                    break
            if has_error == 0:
                # 本轮迭代所有样本都分类正确，终止循环
                break

    def predict(self, X):
        # 每个样本计算出的函数值
        f_value = torch.sum(self.weight * X, dim=1) + self.bias
        # 计算对应的符号函数值，正数为1，负数为-1，0为0
        pred = F.relu(f_value)
        pred[pred == 0] = 1
        return pred


if __name__ == "__main__":
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    model = PerceptronModel(learning_rate=1)
    methods = ["grad", "backward", "optimizer"]
    for method in methods:
        model(X, y, method)
        print("方法:" + method + " 参数为:", model.weight.data, model.bias.data)
