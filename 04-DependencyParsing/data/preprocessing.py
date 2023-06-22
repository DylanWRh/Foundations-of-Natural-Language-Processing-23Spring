word_vec = {}
for line in open('./glove.6B.100d.txt', encoding='utf-8').readlines():
    sp = line.strip().split()
    word_vec[sp[0]] = sp[1:]

words = []
for line in open('./train.conll').readlines():
    sp = line.strip().split('\t')
    if len(sp) == 10:
        if '-' not in sp[0]:
            words.append(sp[1])

words = set(words)

with open('./newglove.6B.100d.txt', 'w', encoding='utf-8') as f:
    for word in words:
        if word in word_vec.keys():
            f.write(word + ' ' + ' '.join(word_vec[word]) + '\n')
        elif word.lower() in word_vec.keys():
            f.write(word.lower() + ' ' + ' '.join(word_vec[word.lower()]) + '\n')
        
