import pickle

invalid_tokens = pickle.load(open('../aclImdb/data/invalid_tokens.pkl', 'rb'))
invalid_tokens = {i: True for i in invalid_tokens}

dataset = ['train.txt', 'test.txt']
for dataset_path in dataset:
    output = open('../aclImdb/data/_'+dataset_path, 'w', encoding='utf-8')
    with open('../aclImdb/data/'+dataset_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words = line[:-1].split(" ")
            valid_tokens = []
            for word in words:
                try:
                    if invalid_tokens[word]:
                        continue
                except KeyError:
                    valid_tokens.append(word)
            output.write(" ".join(valid_tokens) + '\n')
        output.close()
