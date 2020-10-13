import torch
import pickle

word_vectors = pickle.load(open('../aclImdb/data/word_vectors.pkl', 'rb'))
print(word_vectors['for'].shape)
vector = torch.empty(300)
torch.nn.init.uniform_(vector)
word_vectors["<PAD>"] = vector

pickle.dump(word_vectors, open('../aclImdb/data/word_vectors_.pkl', 'wb'))