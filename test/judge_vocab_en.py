import pickle

tot_vocab = dict()
sub_vocab = []


def get_word(line):
    res = ''
    for i in line:
        if i != " ":
            res += i
        else:
            break

    return res


with open('../aclImdb/data/glove.840B.300d.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    length = len(lines)
    cnt = 0
    for line in lines:
        if cnt % 10000 == 0:
            print("{}%".format(cnt/length*100))
        tot_vocab[get_word(line)] = True
        cnt += 1

with open('../aclImdb/data/_vocab.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    length = len(lines)
    cnt = 0
    for line in lines:
        if cnt % 10000 == 0:
            print("{}%".format(cnt/length*100))
        sub_vocab.append(line[:-1])
        cnt += 1

left = []
for i in sub_vocab:
    try:
        if tot_vocab[i]:
            continue
    except KeyError:
        left.append(i)

print(len(left))
print(left)

pickle.dump(left, open('../aclImdb/data/invalid_tokens.pkl', 'wb'))
