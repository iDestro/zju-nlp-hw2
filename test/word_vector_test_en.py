import pickle

word_vector = pickle.load(open('../aclImdb/data/word_vectors.pkl', 'rb'))

no = []
with open('../aclImdb/data/_vocab.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        word = line[:-1]
        try:
            word_vector[word]
        except KeyError:
            no.append(word)

print(no)