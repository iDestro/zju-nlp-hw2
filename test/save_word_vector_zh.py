import pickle

word_vector = pickle.load(open('../THUCNews/data/word_vectors_.pkl', 'rb'))

valid_word_vector = dict()
no = []
with open('../THUCNews/data/_vocab.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(len(lines))
    for line in lines:
        word = line[:-1]
        try:
            word_vector[word]
        except KeyError:
            no.append(word)
        else:
            valid_word_vector[word] = word_vector[word]
print(len(valid_word_vector))
print(no)

pickle.dump(valid_word_vector, open('../THUCNews/data/word_vectors__.pkl', 'wb'))