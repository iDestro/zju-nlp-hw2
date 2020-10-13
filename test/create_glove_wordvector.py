import torch
import pickle


def get_word(line):
    res = ''
    for i in line:
        if i != " ":
            res += i
        else:
            break

    return res


word_vector = dict()
valid_words = dict()

with open('../aclImdb/data/_vocab.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        valid_words[line[:-1]] = True

with open('../aclImdb/data/glove.840B.300d.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    length = len(lines)
    cnt = 0
    for line in lines:
        if cnt % 1000 == 0:
            print("{}%".format(cnt / length * 100))
        word = get_word(line)
        try:
            if valid_words[word]:
                vector = torch.Tensor([float(i) for i in line[:-1].split(" ")[1:]])
                word_vector[word] = vector
        except KeyError:
            pass
        cnt += 1

pickle.dump(word_vector, open('../aclImdb/data/word_vector.pkl', 'wb'))
