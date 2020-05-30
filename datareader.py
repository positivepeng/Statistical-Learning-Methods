#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/5/14 21:17
@author: phil
"""

# 参考：https://xinancsd.github.io/MachineLearning/mnist_parser.html

import numpy as np
import struct
import os

def decode_idx3_ubyte(idx3_ubyte_file):
    with open(idx3_ubyte_file, 'rb') as f:
        print('解析文件：', idx3_ubyte_file)
        fb_data = f.read()

    offset = 0
    fmt_header = '>iiii'    # 以大端法读取4个 unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
    print('魔数：{}，图片数：{}'.format(magic_number, num_images))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(num_rows * num_cols) + 'B'

    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        im = struct.unpack_from(fmt_image, fb_data, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    with open(idx1_ubyte_file, 'rb') as f:
        print('解析文件：', idx1_ubyte_file)
        fb_data = f.read()

    offset = 0
    fmt_header = '>ii'  # 以大端法读取两个 unsinged int32
    magic_number, label_num = struct.unpack_from(fmt_header, fb_data, offset)
    print('魔数：{}，标签数：{}'.format(magic_number, label_num))
    offset += struct.calcsize(fmt_header)
    labels = []

    fmt_label = '>B'    # 每次读取一个 byte
    for i in range(label_num):
        labels.append(struct.unpack_from(fmt_label, fb_data, offset)[0])
        offset += struct.calcsize(fmt_label)
    return np.array(labels)


if __name__=="__main__":
    t10k_images = r"G:\Desktop\t10k-images.idx3-ubyte"
    t10k_labels = r"G:\Desktop\t10k-labels.idx1-ubyte"
    images = decode_idx3_ubyte(t10k_images)
    labels = decode_idx1_ubyte(t10k_labels)
    print(images.shape, labels.shape)

    save_base_dir = r"F:\pycharm_projects\Statistical-Learning-Methods\MNIST_data"
    np.save(os.path.join(save_base_dir, "test_images.npy"), images)
    np.save(os.path.join(save_base_dir, "test_labels.npy"), labels)