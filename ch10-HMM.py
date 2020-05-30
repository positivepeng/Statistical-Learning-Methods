#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# phil : 2019/12/18 15:44
"""
from functools import reduce
import numpy as np
from sklearn.model_selection import train_test_split


class HMM:
    """
        隐马尔可夫模型
    """

    def __init__(self, hidden_state_num, output_num, trans_matrix=None, output_matrix=None, init_prob=None):
        """
        @param hidden_state_num:    隐状态可能的取值的数量
        @param output_num:          输出值可能的取值的数量
        @param trans_matrix:        转移矩阵 (hidden_state_num, hidden_state_num)
        @param output_matrix:       由隐状态映射到输出值的矩阵 (hidden_state_num, output_num)
        @param init_prob:           各个隐状态作为初始状态的概率 (1, hidden_state_num)
        """
        self.hidden_state_num = hidden_state_num
        self.output_num = output_num
        self.trans_matrix = trans_matrix
        self.output_matrix = output_matrix
        self.init_prob = init_prob

    def sequence_probability(self, output_sequence):
        """
        输入观察到的输出序列，输出该序列出现的概率
        @param print_out:           是否输出中间结果
        @param output_sequence:     观察到的序列，也就是输出序列
        @return:                    output_sequence出现的概率
        """
        # 第一个输出值出现的概率
        # 这里的概率是一个向量，prob[i]表示当前停在状态i且状态i对应的输出为output_sequence[i]的概率
        # 这里的prob就是初始隐状态向量乘以该状态输出第一个输出值的概率
        prob = self.init_prob * self.output_matrix[:, output_sequence[0]].T

        # 遍历剩余的序列
        for word in output_sequence[1:]:
            # 计算这个输出值对应的停在各个隐状态的概率
            word_prob = np.zeros_like(prob)
            # 遍历每一个可能的隐状态
            for curr in range(self.hidden_state_num):
                # 对于每一个隐状态，它的前一个状态有可能是任何隐状态
                for before in range(self.hidden_state_num):
                    # 确定了当前隐状态和前一个隐状态，根据转移和输出计算概率
                    word_prob[curr] += prob[before] * self.trans_matrix[before][curr] * self.output_matrix[curr][word]
            # 更新概率值，注意这里的prob的含义，表示当前停在各个隐状态的概率
            prob = word_prob
        # 最后迭代完成后返回总的概率值
        return prob.sum()

    def fit_maxlikelihood(self, sents, tags, k=1):
        # 初始化转移矩阵和输出矩阵
         # 这里使用add-k smoothing进行平滑处理，解决概率连乘中可能出现的0
        self.trans_matrix = np.ones((self.hidden_state_num, self.hidden_state_num)) * k
        self.output_matrix = np.ones((self.hidden_state_num, self.output_num)) * k
        self.init_prob = np.ones(self.hidden_state_num) * k

        # 遍历训练数据
        for sent, tag in zip(sents, tags):
            self.init_prob[tag[0]] += 1.0
            for i in range(len(sent) - 1):
                self.trans_matrix[tag[i]][tag[i + 1]] += 1.0
                self.output_matrix[tag[i]][sent[i]] += 1.0
            # 最后的一个单词只有输出没有转移
            self.output_matrix[tag[-1]][sent[-1]] += 1.0
        # 各个矩阵表示的含义是概率，所以这里需要归一化
        self.init_prob = self.init_prob / np.sum(self.init_prob)

        # 对于转移矩阵和输出矩阵，每一行的和应该为1，表示该隐状态转移到各个隐状态的概率值
        self.trans_matrix = self.trans_matrix / np.sum(self.trans_matrix, axis=1, keepdims=True)
        self.output_matrix = self.output_matrix / np.sum(self.output_matrix, axis=1, keepdims=True)

    def predict(self, output_sequences):
        ans = []
        for seq in output_sequences:
            _, output = self.viterbi_decode(seq)
            ans.append(output)
        return ans

    def viterbi_decode(self, output_sequence):
        # 已知转移矩阵和输出矩阵，求输出序列对应的概率最大的隐状态序列，也就是对输出进行解码
        # 第一步产生的概率向量，各个隐状态初始化概率乘上各个状态输出output_sequence[0]的概率
        # prob[i]表示当前停在隐状态i且到目前为止前面的输出均满住output_sequence
        prob = self.init_prob * self.output_matrix[:, output_sequence[0]].T
        # 记录每一步产生的概率向量
        probs = [prob]
        # 记录prob中每个值来自于上一个向量的哪个值，用于解码，由于第一步并不来自于哪个值，所以这里初始化为-1，便于后面解码
        paths = [[-1 for i in range(self.hidden_state_num)]]
        # 遍历剩下的输出序列
        for i in range(1, len(output_sequence)):
            # 计算当前值对应的概率向量
            new_prob = np.zeros_like(prob)
            # 记录计算当前概率向量时各个元素的值来源于上一个概率向量的哪个值
            new_path = np.zeros_like(prob)
            # 遍历所有可能的状态
            for j in range(self.hidden_state_num):
                # 计算当前的概率向量，假设当前隐状态停在j
                # 上一步的任何一个隐状态都有可能转移到j，计算各个隐状态转移到j的概率
                # 计算上一步得到的概率的最大值再乘隐状态j输出output_sequecne[i]作为这一步停在状态j的概率
                # 这里为了避免连乘造成数值下溢，通过取对数将乘法转换为加法
                new_prob[j] = np.max(prob + np.log(self.trans_matrix[:, j].T)) * self.output_matrix[j, output_sequence[i]]
                # 记录new_prob[j]取最大值时前一个隐状态
                new_path[j] = np.argmax(prob + np.log(self.trans_matrix[:, j].T))
            # 记录概率向量prob各个值取该值时上一个隐状态的序号
            paths.append(new_path)
            # 更新状态向量，用于下一步计算
            prob = new_prob
            # 记录这一步的概率向量
            probs.append(prob)
        # 回溯求解最大概率的序列，最后一个隐状态是最后得到的概率向量的最大值的下标
        ans_seq = [int(np.argmax(prob))]
        # 已知循环直到遇到-1，也就是paths的第一个元素
        while True:
            # ans_seq每次新增都是加在头部
            # paths[-1][temp]的含义，paths是记录的各个tag对应的停在各个隐状态取该值时前一个隐状态的值
            # 每一次处理完成后都会pop，所以这里只需将ans_seq最新插入的值，拿去取paths最后一个元素对应处的隐状态的值
            temp = ans_seq[0]
            if paths[-1][temp] == -1:
                break
            ans_seq = [int(paths[-1][temp])] + ans_seq
            paths.pop()
        # 最后返回每一步的概率向量组成的列表和最大概率的隐状态序列
        return probs, np.array(ans_seq)


if __name__ == "__main__":
    source_path = r"F:\NLP\NER\NER_corpus_chinese-master\人民日报2014NER数据\source_BIO_2014_cropus.txt"
    target_path = r"F:\NLP\NER\NER_corpus_chinese-master\人民日报2014NER数据\target_BIO_2014_cropus.txt"
    sent_num = 2000
    with open(source_path, "r", encoding="utf-8") as f:
        sents = list(map(lambda x: x.strip().split(" "), f.readlines()[:sent_num]))
    with open(target_path, "r", encoding="utf-8") as f:
        tags = list(map(lambda x: x.strip().split(" "), f.readlines()[:sent_num]))

    vocab = list(set(reduce(lambda x, y: x + y, sents)))
    tag = list(set(reduce(lambda x, y: x + y, tags)))

    word2id = {v:index for index, v in enumerate(vocab)}
    tag2id = {v:index for index, v in enumerate(tag)}

    print("word2id", len(word2id))
    print("tag2id", len(tag2id))

    def tokenizer(texts, id_mapper):
        text_ids = []
        for text in texts:
            text_ids.append(list(map(lambda x:id_mapper[x], text)))
        return text_ids

    sent_tokens = tokenizer(sents, word2id)
    tag_tokens = tokenizer(tags, tag2id)

    # print(tag_tokens)
    X_train, X_test, y_train, y_test = train_test_split(sent_tokens, tag_tokens, test_size=0.2, random_state=42)

    model = HMM(hidden_state_num=len(tag), output_num=len(vocab))

    model.fit_maxlikelihood(X_train, y_train)
    print("训练完成")
    y_preds = model.predict(X_test)
    acc = 0.0
    total = 0.0
    for pred, gold in zip(y_preds, y_test):
        acc += np.sum(pred == gold)
        total += len(pred)
    print("accuracy: {} %".format(acc/total*100))

    """
    word2id 2923
    tag2id 9
    训练完成
    accuracy: 87.6248889875666 %
    """
