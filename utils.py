# coding: UTF-8
import torch
import pickle
import time
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader


class Preprocess():
    def __init__(self, config):
        self.config = config
        # 截取一个文档的前sel_len作为训练样本
        self.sen_len = self.config.sen_len
        # 用于保存词和索引的映射关系
        self.idx2word = None
        self.word2idx = None
        # 保存类别标签
        self.categories = ['pos', 'neg'] if config.language == 'en' else ['体育', '财经', '房产', '家居', '教育', '科技', '时尚',
                                                                          '时政', '游戏', '娱乐']
        # 保存类别到索引的映射
        self.cat_to_id = dict(zip(self.categories, range(len(self.categories))))
        # 初始化词和索引的映射关系和载入词向量
        self.init_and_set_embedding()

    # 获取索引到词，词到索引的映射，以及加载词向量矩阵
    def init_and_set_embedding(self):
        word2vec = pickle.load(open(self.config.word_vectors_path, 'rb'))
        self.idx2word = []
        self.word2idx = {}
        embedding_matrix = []
        for i, word in enumerate(word2vec):
            print('get words #{}'.format(i + 1), end='\r')
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            embedding_matrix.append(word2vec[word].numpy())
        print('')
        embedding_matrix = torch.tensor(embedding_matrix)
        print("total words: {}".format(len(embedding_matrix)))
        self.config.set_embedding(embedding_matrix)

    # 将句子变成一样的长度
    def pad_sequence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    # 讲句子变为索引序列
    def sentence_word2idx(self, sentences):
        sentence_list = []
        for i, sen in enumerate(sentences):
            print('sentence count #{}'.format(i + 1), end='\r')
            sentence_idx = []
            for word in sen:
                sentence_idx.append(self.word2idx[word])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    # 标签向量化
    def labels_to_tensor(self, y):
        y = [self.cat_to_id[label] for label in y]
        return torch.LongTensor(y)

    # 分别生成训练，验证和测试的Dataloader
    def get_iters(self):
        paths = [self.config.train_path, self.config.dev_path, self.config.test_path]
        iters = []
        for path in paths:
            x, y = load_data(path, self.config.language)
            x = self.sentence_word2idx(x)
            y = self.labels_to_tensor(y)
            dataset = TextDataset(x, y)
            iter = DataLoader(dataset,
                              shuffle=True,
                              batch_size=self.config.batch_size,
                              num_workers=0)
            iters.append(iter)
        return iters


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# 载入数据
def load_data(path, language):
    if language == 'zh':
        start_index = 3
    else:
        start_index = 4
    with open(path, 'r', encoding='utf-8') as f:
        x = []
        y = []
        for line in f.readlines():
            if language == 'zh':
                x.append(list(line[start_index:-1]))
            else:
                x.append(line[start_index:-1].split(" "))
            y.append(line[:start_index - 1])
    return x, y


# 文本数据类
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.lable = y

    def __len__(self):
        return len(self.lable)

    def __getitem__(self, index):
        return self.data[index], self.lable[index]
