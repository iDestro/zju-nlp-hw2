import torch
import numpy as np
import pickle
word2vec = {}
with open('../THUCNews/data/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5', 'r', encoding='utf-8') as f:
    head = f.readline()
    next(f)
    corpus_length, dim = head.split(" ")
    for line in f.readlines():
        temp = line.split(" ")
        word2vec[temp[0]] = torch.Tensor([float(i) for i in temp[1:-1]])
    pickle.dump(word2vec, open('../THUCNews/data/word_vectors.pkl', 'wb'))
