#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/2/20 23:33
@author: phil
"""
from functools import reduce
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim


class Embedding_CRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_of_labels, start_tag_id, end_tag_id, pad_tag_id):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.CRF = CRF(num_of_labels, start_tag_id, end_tag_id, pad_tag_id)
        self.embed2label = nn.Linear(embed_dim, num_of_labels)

    def forward(self, x, tags, mask):
        embed_output = self.embed(x)
        emissions = self.embed2label(embed_output)
        return self.CRF(emissions, tags, mask)

    def predict(self, x, mask):
        embed_output = self.embed(x)
        emissions = self.embed2label(embed_output)
        return self.CRF.viterbi_decode(emissions, mask)


class CRF(nn.Module):
    """线性链条件随机场"""

    def __init__(self, num_of_labels, start_tag_id, end_tag_id, pad_tag_id=None, trans=None):
        """
        num_of_label: 标签数量
        start_tag_id: 开始标签的id
        end_tag_id: 结束标签的id
        pad_tag_id: 填充标签的id
        trans: 自定义转移矩阵（用于测试）
        """
        super(CRF, self).__init__()
        self.num_of_labels = num_of_labels
        self.start_tag_id = start_tag_id
        self.end_tag_id = end_tag_id
        self.pad_tag_id = pad_tag_id

        # 定义转移矩阵，shape(num_of_labels, num_of_labels)
        if trans is not None:
            self.transitions = nn.Parameter(trans)
        else:
            self.transitions = nn.Parameter(self.init_weights())

    def init_weights(self):
        # 初始化转移矩阵为-1到1的均匀分布
        trans = torch.randn(self.num_of_labels, self.num_of_labels)
        nn.init.uniform_(trans, -0.1, 0.1)

        # # 关于<start>的特殊情况，所有标签都不能转移到<start>
        trans[:, self.start_tag_id] = -10000.0

        # # 关于<end>的特殊情况，不能由<end>转移到任何其他标签
        trans[self.end_tag_id, :] = -10000.0

        if self.pad_tag_id is not None:
            # <pad>标签前后，要么跟<pad>，要么跟<end>，不能跟其他标签
            trans[self.pad_tag_id, :] = -10000.0
            trans[:, self.pad_tag_id] = -10000.0
            trans[self.pad_tag_id, self.pad_tag_id] = 0
            trans[self.pad_tag_id, self.end_tag_id] = 0
        return trans

    def forward(self, emissions, tags, mask=None):
        """
        emissions: 发射得分 shape(batch_size, seq_len, num_of_labels), 表示每个单词被标记为各个标签的得分
        tags： 真实标签序列  shape(batch_size, seq_len), 句子对应的真实标签序列
        """
        nll = -1 * self.log_likelihoods(emissions, tags, mask=None)
        return nll

    def log_likelihoods(self, emissions, tags, mask=None):
        if mask is None:
            # 所有单词都计算得分
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        # 真实标签的得分，emissions score + trans score
        scores = self.compute_scores(emissions, tags, mask=mask)

        # 所有可能的标签序列的得分的指数然后再求对数的结果
        partition = self.compute_log_partition(emissions, mask=mask)

        return torch.sum(scores - partition)

    def compute_scores(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        scores = torch.zeros(batch_size)

        # batch中所有句子的开始单词
        first_tags = tags[:, 0]

        # batch中所有句子的结束单词
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        # 转移得分，从<start>标签转移到tag[0]的的得分
        t_scores = self.transitions[self.start_tag_id, first_tags]

        # 计算第一个tag的emission score
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()

        # 总得分=发射得分+转移得分
        scores += e_scores + t_scores

        # 循环遍历剩下的单词
        for i in range(1, seq_len):
            # batch中所有句子第i个单词是否是pad决定该单词是否有效
            is_valid = mask[:, i]

            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]

            # 计算emission score
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            # 计算转移得分
            t_scores = self.transitions[previous_tags, current_tags]

            # 保留不是pad的部分
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        # 添加由tag[-1]到<end>标签的转移得分
        scores += self.transitions[last_tags, self.end_tag_id]

        return scores

    def compute_log_partition(self, emissions, mask):
        """
            使用前向算法计算所有可能的标签序列的得分的和的对数
            emissions: shape (batch_size, seq_len, nb_labels)
            mask : shape (batch_size, seq_len)
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # 计算由开始标签转移到各个标签的得分+发射得分
        # alphas[j] 表示当前停在标签j的可能的序列的得分
        alphas = self.transitions[self.start_tag_id, :].unsqueeze(0) + emissions[:, 0]  # shape (batch_size, nb_labels)

        # 计算剩下的单词
        for i in range(1, seq_length):
            alpha_t = []

            # 遍历每一个标签
            for tag in range(nb_labels):
                # 每一个句子对应的第i个单词对应到tag的emission score
                e_scores = emissions[:, i, tag]
                # (batch_size, 1)
                e_scores = e_scores.unsqueeze(1)

                # 上一个tag可以是任意一种标签，转移到当前tag的得分
                t_scores = self.transitions[:, tag]
                # (1, nb_labels)
                t_scores = t_scores.unsqueeze(0)

                # 这里为什么能够直接相加 见博客
                scores = e_scores + t_scores + alphas

                # 这里为什么计算logsumexp 见博客
                # (batch_size)
                alpha_t.append(torch.logsumexp(scores, dim=1))

            # (batch_size, nb_labels)
            new_alphas = torch.stack(alpha_t).t()

            # 如果当前单词不是pad则更新alpha的值，否则保留原alpha的值
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

        # 加上转移到<end>的得分
        last_transition = self.transitions[:, self.end_tag_id]
        end_scores = alphas + last_transition.unsqueeze(0)

        # 最后再求logsumexp
        return torch.logsumexp(end_scores, dim=1)

    def viterbi_decode(self, emissions, mask):
        """
        维特比解码，使用动态规划算法计算得分最高的预测序列
        """
        batch_size, seq_length, nb_labels = emissions.shape
        alphas = self.transitions[self.start_tag_id, :].unsqueeze(0) + emissions[:, 0]  # shape (batch_size, nb_labels)

        # 记录每一步转移得分最高的来源
        backpointers = []

        for i in range(1, seq_length):
            alpha_t = []
            backpointers_t = []
            for tag in range(nb_labels):
                e_scores = emissions[:, i, tag]
                # (batch_size, 1)
                e_scores = e_scores.unsqueeze(1)

                t_scores = self.transitions[:, tag]
                # (1, nb_labels)
                t_scores = t_scores.unsqueeze(0)

                # scores (batch_size, nb_labels)
                # 各个标签转移到当前tag的得分
                scores = e_scores + t_scores + alphas

                max_score, max_score_tag = torch.max(scores, dim=-1)

                alpha_t.append(max_score)

                backpointers_t.append(max_score_tag)

            # (batch_size, nb_labels)
            new_alphas = torch.stack(alpha_t).t()

            # 如果当前单词不是pad则更新alpha的值，否则保留原alpha的值
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

            backpointers.append(backpointers_t)

        # 加上转移到<end>的得分
        last_transition = self.transitions[:, self.end_tag_id]
        end_scores = alphas + last_transition.unsqueeze(0)

        # 最终的最高得分，以及最高得分对应的最终的tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):
            sample_length = emission_lengths[i].item()  # batch中第i个句子的长度
            sample_final_tag = max_final_tags[i].item()  # 第i个句子最高得分对应的标签
            sample_backpointers = backpointers[: sample_length - 1]  # 去除backpointers中pad部分的信息

            # 从得分最高的序列对应的最后一个标签向前回溯
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # 记录最优序列
            best_sequences.append(sample_path)

        return -max_final_scores, best_sequences

    def _find_best_path(self, sample_id, final_tag, backpointers):
        # index表示第index个句子
        # final_tag是回溯的开始标签
        # backpoointers记录了从前向后计算过程中最高得分的来源
        best_path = [final_tag]

        for backpointers_t in reversed(backpointers):
            best_tag = backpointers_t[final_tag][sample_id].item()
            best_path.insert(0, best_tag)

        return best_path


if __name__ == "__main__":
    torch.manual_seed(2020)

    # 读入句子
    # source_path = r"F:\NLP\NER\NER_corpus_chinese-master\人民日报2014NER数据\source_BIO_2014_cropus.txt"
    # target_path = r"F:\NLP\NER\NER_corpus_chinese-master\人民日报2014NER数据\target_BIO_2014_cropus.txt"
    # sent_num = 2
    # with open(source_path, "r", encoding="utf-8") as f:
    #     sents = list(map(lambda x: x.strip().split(" "), f.readlines()[:sent_num]))
    # with open(target_path, "r", encoding="utf-8") as f:
    #     tags = list(map(lambda x: x.strip().split(" "), f.readlines()[:sent_num]))

    sents = ["人 民 网 1 月 1 日 讯 据 《 纽 约 时 报 》 报 道 , 美 国 华 尔 街 股 市 在 2 0 1 3 年 的 最 后 一 天 继 续 上 涨 , 和 全 球 股 市 一 样 , "
             "都 以 最 高 纪 录 或 接 近 最 高 纪 录 结 束 本 年 的 交 易 。 "
             "《 纽 约 时 报 》 报 道 说 , 标 普 5 0 0 指 数 今 年 上 升 2 9 . 6 % , 为 1 9 9 7 年 以 来 的 最 大 涨 幅 ; 道 琼 斯 工 业 平 均 指 数 上 升 "
             "2 6 . 5 % , 为 1 9 9 6 年 以 来 的 最 大 涨 幅 ; 纳 斯 达 克 上 涨 3 8 . 3 % 。"]
    tags = ["O O O B_T I_T I_T I_T O O O B_LOC I_LOC O O O O O O B_LOC I_LOC I_LOC I_LOC I_LOC O O O B_T I_T I_T I_T "
            "I_T O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O "
            "O B_LOC I_LOC O O O O O O O O O O O O O O B_T I_T O O O O O O O O O B_T I_T I_T I_T I_T O O O O O O O O "
            "B_ORG I_ORG I_ORG O O O O O O O O O O O O O O O B_T I_T I_T I_T I_T O O O O O O O O B_PER I_PER O O O O "
            "O O O O O O"]

    # 对数据进行预处理
    vocab = ["<unk>", "<pad>", "<bos>", "<eos>"] + list(set(reduce(lambda x, y: x + y, sents)))
    WORD_PAD_ID = 1

    tag = ["<pad>", "<bos>", "<eos>"] + list(set(reduce(lambda x, y: x + y, tags)))
    TAG_PAD_ID = 0
    TAG_BEGIN_ID = 1
    TAG_END_ID = 1

    word2id = {v: index for index, v in enumerate(vocab)}
    tag2id = {v: index for index, v in enumerate(tag)}

    if len(word2id) < 100:
        print("word2id", word2id)
        print("tag2id", tag2id)


    def tokenizer(texts, id_mapper):
        text_ids = []
        for text in texts:
            text_ids.append(list(map(lambda x: id_mapper[x], text)))
        return text_ids


    train_sents, test_sents, train_tags, test_tags = train_test_split(sents, tags, test_size=0.2, random_state=42)

    sent_tokens = tokenizer(sents, word2id)
    tag_tokens = tokenizer(tags, tag2id)
    lens = [len(tag_list) for tag_list in tags]
    max_sent_size = max(lens)
    print("Max sentence size:", max_sent_size)

    training_data = []
    for i in range(len(sent_tokens)):
        training_data.append((sent_tokens[i], tag_tokens[i]))

    x_sent = torch.full((len(sent_tokens), max_sent_size), WORD_PAD_ID, dtype=torch.long)
    x_tags = torch.full((len(sent_tokens), max_sent_size), TAG_PAD_ID, dtype=torch.long)

    for i in range(len(lens)):
        x_sent[i, :lens[i]] = torch.tensor(sent_tokens[i])
        x_tags[i, :lens[i]] = torch.tensor(tag_tokens[i])

    mask = (x_tags != TAG_PAD_ID).float()

    id2tag = {id: tag for tag, id in tag2id.items()}

    model = Embedding_CRF(vocab_size=len(word2id), embed_dim=10, num_of_labels=len(tag2id),
                          start_tag_id=TAG_BEGIN_ID, end_tag_id=TAG_END_ID, pad_tag_id=TAG_PAD_ID)

    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(100):  # normally you would NOT do 300 epochs, it is toy data

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, turn them into Tensors of word indices.
        # Step 3. Run our forward pass.
        loss = model(x_sent, x_tags, mask=mask)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

        # Check predictions before training
        if epoch % 10 == 0:
            with torch.no_grad():
                loss = model(x_sent, x_tags, mask=mask)
                print(epoch, "loss", loss)

    print('Predictions after training:')
    right_cnt = 0
    total_cnt = 0
    with torch.no_grad():
        scores, seqs = model.predict(x_sent, mask=mask)
        for score, seq, true_tags in zip(scores, seqs, tags):
            pred_list = tokenizer([seq], id2tag)[0]
            for pred, gold in zip(pred_list, true_tags):
                if pred == gold:
                    right_cnt += 1
                total_cnt += 1
    print("正确率", right_cnt * 1.0 / total_cnt)
