from collections import Counter
import numpy as np
import datetime
from dataloader import load_all_num
import time

# compute_distances_***_loops 参考 http://cs231n.stanford.edu/

class KNN:
    """基于欧式距离的KNN模型"""
    def __init__(self):
        pass

    def fit(self, X, y):
        # 没有显式的训练过程，只需要把数据记录下来
        self.X_train = X  # X的每一行表示一个训练样本
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        # 下面是三种距离计算方式，num_loops是表示使用几重循环计算距离矩阵
        # dists是距离矩阵，dists[i][j]表示self.X（训练集）中的第i个样本和X（测试集）的第j个样本间的距离
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        # 有了距离矩阵后就可进行预测，返回测试集样本的最近的k个样本中出现次数最多的标签
        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        # 使用两重循环计算距离矩阵
        num_test = X.shape[0]  # 测试集样本个数
        num_train = self.X_train.shape[0]  # 训练集样本个数
        dists = np.zeros((num_test, num_train))  # 距离矩阵

        # 记录开始时间
        start = datetime.datetime.now()
        # 两重循环计算距离矩阵，计算每个测试样本到各个训练样本间的距离
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.sqrt(np.sum((X[i] - self.X_train[j]) ** 2))
        return dists

    def compute_distances_one_loop(self, X):
        # 使用一重循环计算距离矩阵
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # 原理：对于dists的每一行，dists[i]代表的X[i]到X_train的每个样本间的距离
        # dists[i][j]**2 = (X[i] - X_train[j])**2 = X[i]**2 - 2*X[i]*X_train[j] + X_train[j]**2
        # 第一部分：X[i]**2，直接算，结果为标量
        # 第二部分：2*X[i]*X_train[j]，目标是求dists[i]，所以这个部分结果应当是一个一维矩阵
        #          这里使用X_train的转置（一列表示一个样本），使用矩阵乘，X[i]的每一行乘X_train的每一列
        # 第三部分：X_train[j]**2，同上这个返回也应当是一个矩阵，所以直接使用X_train*X_train然后按行求和
        for i in range(num_test):
            dists[i] = np.sqrt(
                (X[i] * X[i]).sum() - 2 * X[i].dot(self.X_train.T) + (self.X_train * self.X_train).sum(axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        # 不使用循环计算距离矩阵
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # np.tile(array, shape)将array看作一个元素，扩展为一个形状为shape的新元素
        # 原理：这里直接求dists，shape为（num_test, num_train）
        # 依然是上面的三个部分，不过这里使用np.tile函数直接为dists中的每个元素构造好三个部分
        # np.tile(np.sum(X * X, axis=1), (num_train, 1)).T
        #   将X*X按行求和，每个元素表示sum(X[i]*X[i])，形状为(1, num_test)
        #   将上面得到的矩阵作为元素扩展为一个形状为(num_train, 1)的矩阵
        #   最后得到一个(num_train, num_test)的矩阵，然后转置shape为(num_test, num_train)
        # np.tile(np.sum(self.X_train * self.X_train, axis=1), (num_test, 1))
        #   原理同上，只不过这里不转置，最终shape为(num_test, num_train)
        # 2 * X.dot(self.X_train.T)
        #   X 每一行表示一个样本
        #   X_train每一行表示一个样本，转置后每一列为一个样本
        #   矩阵乘后得到一个shape为(num_test, num_train)的矩阵，每个元素表示对应的矩阵的内积
        dists = np.sqrt(np.tile(np.sum(X * X, axis=1), (num_train, 1)).T + \
                        np.tile(np.sum(self.X_train * self.X_train, axis=1), (num_test, 1)) - \
                        2 * X.dot(self.X_train.T))
        return dists

    def predict_labels(self, dists, k=1):
        # 根据距离矩阵为每个待预测样本预测标签
        # dists[i]表示的第i个测试样本和各个训练样本之间的距离
        num_test = dists.shape[0]
        # 用于保存测试集预测结果
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            # 得到离第i个测试样本距离最近的k个样本点的编号
            closest_y_ids = list(np.argsort(dists[i])[:k])
            # 根据得到的样本编号得到对应的y值
            for y_id in closest_y_ids:
                closest_y.append(self.y_train[y_id][0])  # closest_y中元素必须是int
            # 找出最近的k个y值中出现次数最多的y
            y_id_label_counter = Counter(closest_y)
            y_pred[i] = y_id_label_counter.most_common(1)[0][0]
        return y_pred

    def score(self, X, y, k=1):
        pred = self.predict(X, k=k)
        return np.sum(pred == y.reshape(pred.shape)) / len(y)


def sklearn_test(train_X, train_y, test_X, test_y, n_neighbors=1):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors)
    model.fit(train_X, train_y)
    return model.score(test_X, test_y)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_all_num(test_size=500)

    print("训练集规模", X_train.shape)
    print("测试集规模", X_test.shape)

    start = time.time()
    model = KNN()
    model.fit(X_train, y_train)
    print("手写KNN验证集精度 %.2f" % model.score(X_test, y_test, k=5))
    end = time.time()

    print("过程耗时 %.2f" % (end - start))
    print("sklearn KNN验证集精度 %.2f" % sklearn_test(X_train, y_train[:, 0], X_test, y_test[:, 0], n_neighbors=5))

    """
    训练集规模 (60000, 784)
    测试集规模 (500, 784)
    手写KNN验证集精度 0.97
    过程耗时 6.90
    sklearn KNN验证集精度 0.97
    """
