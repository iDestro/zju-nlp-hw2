
words = []
with open('../aclImdb/data/_train.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        words += list(line[4:-1].split(" "))

with open('../aclImdb/data/_test.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        words += list(line[4:-1].split(" "))

s = set(words)

with open('../aclImdb/data/_vocab.txt', 'w', encoding='utf-8') as f:
    for word in s:
        f.write(word+'\n')


