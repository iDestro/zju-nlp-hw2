# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset


class Preprocess():
    def __init__(self, sentences, sen_len, idx2word, word2idx):
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.embedding_matrix = []
        self.categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        self.cat_to_id = dict(zip(self.categories, range(len(self.categories))))

    def pad_sequence(self, sentence):
        # 將每個句子變成一樣的長度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                sentence_idx.append(self.word2idx[word])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        y = [self.cat_to_id[label] for label in y]
        return torch.LongTensor(y)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_data(path):
    # 把 training 時需要的 data 讀進來
    # 如果是 'training_label.txt'，需要讀取 label，如果是 'training_nolabel.txt'，不需要讀取 label
    with open(path, 'r', encoding='utf-8') as f:
        x = []
        y = []
        for line in f.readlines():
            x.append(list(line[3:-1]))
            y.append(line[:2])
    return x, y


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.lable = y

    def __len__(self):
        return len(self.lable)

    def __getitem__(self, index):
        return self.data[index], self.lable[index]
