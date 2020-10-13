import random

train = open('../aclImdb/data/train.txt', 'w', encoding='utf-8')
dev = open('../aclImdb/data/dev.txt', 'w', encoding='utf-8')

with open('../aclImdb/data/_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    random.shuffle(lines)
    length = len(lines)
    split_point = int(0.2*length)
    cnt = 0
    for line in lines:
        if cnt < split_point:
            train.write(line)
        else:
            dev.write(line)
        cnt += 1
train.close()
dev.close()