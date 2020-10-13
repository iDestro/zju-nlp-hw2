import pickle

word_vector = pickle.load(open('../THUCNews/data/word_vectors_.pkl', 'rb'))

no = []
with open('../THUCNews/data/_vocab.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        word = line[:-1]
        try:
            word_vector[word]
        except KeyError:
            no.append(word)

print(no)