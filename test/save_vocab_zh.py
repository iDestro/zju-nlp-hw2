
words = []
with open('../THUCNews/data/train.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        words += list(line[3:-1])

with open('../THUCNews/data/test.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        words += list(line[3:-1])

with open('../THUCNews/data/dev.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        words += list(line[3:-1])

s = set(words)

with open('../THUCNews/data/_vocab.txt', 'w', encoding='utf-8') as f:
    for word in s:
        f.write(word+'\n')


