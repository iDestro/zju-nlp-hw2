import pickle

vocab = []
with open('../THUCNews/data/vocab_new.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        vocab.append(line[:-1])
    pickle.dump(vocab, open('../THUCNews/data/vocab.pkl', 'wb'))