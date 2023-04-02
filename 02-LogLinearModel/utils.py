from pandas import read_csv
import re
import math, random
from stopwords import stopwords


def read_data(filename):
    csv = read_csv(filename)
    X_lst = csv['data'].tolist()
    y_lst = csv['target'].tolist()
    for i in range(len(X_lst)):
        X_lst[i] = data_clean(X_lst[i])
    return X_lst, y_lst


def data_clean(X):
    punc = '[,.?!<>\n()-\[\]*:;]'
    X = re.sub(punc, ' ', X)
    return X.strip().lower()


def build_vocab(X_lst):
    vocab = {}
    for X in X_lst:
        words = X.split()
        for word in words:
            if len(word) >= 3 and word not in stopwords:
                vocab[word] = vocab.get(word, 0) + 1
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    return vocab


def feature_extraction(X_lst, dim=5000, word_embed=None):
    embed_return = False
    if word_embed is None:
        embed_return = True
        vocab = build_vocab(X_lst)[:dim]
        word_embed = {}
        for i, item in enumerate(vocab):
            word_embed[item[0]] = i

    # using set here is faster than one-hot vector without numpy/torch acceleration
    # the elements in each set represents the indices where value is 1 in the vector
    features = [set() for i in range(len(X_lst))]

    for i, X in enumerate(X_lst):
        words = X.split()
        for word in words:
            if word in word_embed.keys():
                features[i].add(word_embed[word])

    if embed_return:
        return features, word_embed
    return features


def get_features(filename, dim=5000, word_embed=None):
    X_lst, y_lst = read_data(filename)

    if word_embed is None:
        features, word_embed = feature_extraction(X_lst, dim=dim)
        return features, y_lst, word_embed

    features = feature_extraction(X_lst, dim=dim, word_embed=word_embed)
    return features, y_lst


def _shuffle(X_lst, y_lst):
    n = len(X_lst)
    shuffle_lst = list(range(n))
    random.shuffle(shuffle_lst)
    X_shuffled = [X_lst[i] for i in shuffle_lst]
    y_shuffled = [y_lst[i] for i in shuffle_lst]
    return X_shuffled, y_shuffled


def argmax(lst):
    idx = 0
    max_val = lst[0]
    for i in range(1, len(lst)):
        if lst[i] > max_val:
            max_val = lst[i]
            idx = i
    return idx 


def softmax(scores):
    max_score = max(scores)
    scores = [score - max_score for score in scores]
    scores = [math.exp(score) for score in scores]
    sum_scores = sum(scores)
    scores = [score / sum_scores for score in scores]
    return scores


def acc(y_pred, y_lst):
    num_correct = 0
    n_test = len(y_lst)
    for i in range(n_test):
        if y_pred[i] == y_lst[i]:
            num_correct += 1
    acc = num_correct / n_test
    return acc


def f1_micro(y_pred, y_lst, n_classes=20):
    table = get_table(y_pred, y_lst)
    TP = [0 for i in range(n_classes)]
    FP = [0 for i in range(n_classes)]
    FN = [0 for i in range(n_classes)]
    for i in range(n_classes):
        TP[i] = table[i][i]
        for j in range(n_classes):
            if j == i:
                continue
            FP[i] += table[j][i]
            FN[i] += table[i][j]
    P = sum(TP) / (sum(TP) + sum(FP))
    R = sum(TP) / (sum(TP) + sum(FN))
    F1 = 2 * P * R / (P + R)
    return F1


def f1_macro(y_pred, y_lst, n_classes=20):
    table = get_table(y_pred, y_lst)
    TP = [0 for i in range(n_classes)]
    FP = [0 for i in range(n_classes)]
    FN = [0 for i in range(n_classes)]
    for i in range(n_classes):
        TP[i] = table[i][i]
        for j in range(n_classes):
            if j == i:
                continue
            FP[i] += table[j][i]
            FN[i] += table[i][j]
    P = [0 for i in range(n_classes)]
    R = [0 for i in range(n_classes)]
    F1 = [0 for i in range(n_classes)]
    for i in range(n_classes):
        P[i] = TP[i] / (TP[i] + FP[i])
        R[i] = TP[i] / (TP[i] + FN[i])
        F1[i] = 2 * P[i] * R[i] / (P[i] + R[i])
    return sum(F1) / n_classes


def get_table(y_pred, y_lst, n_classes=20):
    # table[i][j]: number of cases (y_true=i and y_pred=j)
    table = [[0 for j in range(n_classes)] for i in range(n_classes)]
    for i, j in zip(y_lst, y_pred):
        table[i][j] += 1
    return table

