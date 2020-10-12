import torch
from utils import load_data

x1, _ = load_data('../THUCNews/data/train.txt')
x2, _ = load_data('../THUCNews/data/dev.txt')
x3, _ = load_data('../THUCNews/data/test.txt')

x = [x1, x2, x3]
s = set()

for m in x:
    for i in m:
        for j in i:
            s.add(j)

s.add('<PAD>')

with open('../THUCNews/data/vocab_new.txt', 'w', encoding='utf-8') as f:
    for i in s:
        f.write(i+'\n')





