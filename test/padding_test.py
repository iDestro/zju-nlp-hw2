import math
import matplotlib.pyplot as plt
import collections
Min = 1e9
Max = 0


with open('../THUCNews/data/train.txt', 'r', encoding='utf-8') as f:
    count = 0
    tot = 0
    res = []
    for line in f.readlines():
        cur_len = len(list(line[3:]))
        res.append(cur_len)
        if cur_len == 9:
            print(line)
        Min = min(Min, cur_len)
        Max = max(Max, cur_len)
        tot += cur_len
        count += 1
counter = collections.Counter(res)
for i in res:
    print(i)
plt.hist(res)
plt.show()

print("min: ", Min)
print("max: ", Max)
print("avg: ", tot / count)


