import torch
import pickle

word2vec = pickle.load(open('../THUCNews/data/word_vectors.pkl', 'rb'))

vocab = pickle.load(open('../THUCNews/data/vocab.pkl', 'rb'))


# for i in word2vec.values():
#     print(i.size())

# for i in vocab:
#     if i not in word2vec.keys():
#         vector = torch.empty(300)
#         torch.nn.init.uniform_(vector)
#         word2vec[i] = vector
#
# pickle.dump(word2vec, open('../THUCNews/data/word_vectors_.pkl', 'wb'))


# left = [i for i in vocab if i not in word2vec.keys()]
# print(left)

# for i, d in enumerate(word2vec):
#     print(d)
#     print(word2vec[d])