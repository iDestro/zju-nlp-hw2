# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
from TextCNN import Config, Model
from utils import TextDataset, Preprocess, get_time_dif, load_data
import pickle
from torch.utils.data import DataLoader
import multiprocessing

if __name__ == '__main__':
    dataset = './THUCNews'  # 数据集

    word2vec = pickle.load(open('./THUCNews/data/word_vectors_.pkl', 'rb'))
    idx2word = []
    word2idx = {}
    embedding_matrix = []
    for i, word in enumerate(word2vec):
        print('get words #{}'.format(i + 1), end='\r')
        word2idx[word] = len(idx2word)
        idx2word.append(word)
        embedding_matrix.append(word2vec[word].numpy())
    print('')
    embedding_matrix = torch.tensor(embedding_matrix)
    print("total words: {}".format(len(embedding_matrix)))

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    config = Config(dataset)
    start_time = time.time()
    print("Loading data...")
    train_x, train_y = load_data(config.train_path)
    val_x, val_y = load_data(config.dev_path)
    test_x, test_y = load_data(config.test_path)

    train_data_preprocess = Preprocess(train_x, 600, idx2word, word2idx)
    val_data_preprocess = Preprocess(val_x, 600, idx2word, word2idx)
    test_data_preprocess = Preprocess(test_x, 600, idx2word, word2idx)

    config.set_embedding(embedding_matrix)

    train_x = train_data_preprocess.sentence_word2idx()
    train_y = train_data_preprocess.labels_to_tensor(train_y)

    val_x = val_data_preprocess.sentence_word2idx()
    val_y = val_data_preprocess.labels_to_tensor(val_y)

    test_x = test_data_preprocess.sentence_word2idx()
    test_y = test_data_preprocess.labels_to_tensor(test_y)

    train_dataset = TextDataset(train_x, train_y)
    val_dataset = TextDataset(val_x, val_y)
    test_dataset = TextDataset(test_x, test_y)

    train_iter = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=config.batch_size,
                            num_workers=0)

    dev_iter = DataLoader(val_dataset,
                          shuffle=True,
                          batch_size=config.batch_size,
                          num_workers=0)

    test_iter = DataLoader(test_dataset,
                           shuffle=True,
                           batch_size=config.batch_size,
                           num_workers=0)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
