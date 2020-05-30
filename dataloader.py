#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/2/9 16:13
@author: phil
"""

import numpy as np
import matplotlib.pyplot as plt

train_image_path = "../MNIST_data/train_images.npy"
train_label_path = "../MNIST_data/train_labels.npy"
test_image_path = "../MNIST_data/test_images.npy"
test_label_path = "../MNIST_data/test_labels.npy"


def load_all_num(train_size=60000, test_size=10000):
    train_images = np.load(train_image_path)
    train_labels = np.load(train_label_path)
    train_images = train_images.reshape(60000, 28*28)
    train_labels = train_labels.reshape(60000, 1)

    test_images = np.load(test_image_path)
    test_labels = np.load(test_label_path)
    test_images = test_images.reshape(10000, 28 * 28)
    test_labels = test_labels.reshape(10000, 1)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return train_images[:train_size], train_labels[:train_size], test_images[:test_size], test_labels[:test_size]


def load_01(train_size=60000, test_size=10000):
    train_images, train_labels, test_images, test_labels = load_all_num(train_size, test_size)
    train_mask = (train_labels == 0) + (train_labels == 1)
    train_mask = train_mask[:, 0]
    test_mask = (test_labels == 0) + (test_labels == 1)
    test_mask = test_mask[:, 0]

    return train_images[train_mask], train_labels[train_mask], test_images[test_mask], test_labels[test_mask]


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_all_num()
    print("train shape", train_images.shape)
    print("label shape", train_labels.shape)

    num = 10
    images = [None] * num
    cnt = 0
    for i in range(len(train_images)):
        label = int(train_labels[i][0])
        if images[label] is None:
            images[label] = train_images[i]
            cnt += 1
        if cnt == num:
            break
    for i in range(num):
        plt.subplot(1, 10, i+1)
        plt.imshow(images[i].reshape(28, 28))
    plt.show()

