from utils import argmax, softmax, f1_micro, f1_macro, acc
import random
from sklearn.metrics import accuracy_score as acc_sk
from sklearn.metrics import f1_score as f1_sk


class LogLinear:
    def __init__(self, lr=1e-3, output_channel=20, input_channel=5000, reg=None, momentum=None, seed=42):
        self.lr = lr
        self.output_channel = output_channel
        self.input_channel = input_channel
        self.reg = reg
        self.momentum = momentum

        w_init = 0.1 / input_channel
        random.seed(seed)
        self.w = [[random.random() * w_init for i in range(input_channel)] for j in range(output_channel)]
        self.v = [[0.0 for i in range(input_channel)] for j in range(output_channel)]

    def forward(self, features_lst):
        scores_lst = []
        for features in features_lst:
            scores = []
            for c in range(self.output_channel):
                scores.append(sum([i * self.w[c][i] for i in features]))
            scores = softmax(scores)
            scores_lst.append(scores)
        y_lst = [argmax(scores) for scores in scores_lst]
        return y_lst, scores_lst
    
    def backward(self, features_lst, y_lst):
        y_pred, scores_lst = self.forward(features_lst)
        w_grad = [[0 for i in range(self.input_channel)] for j in range(self.output_channel)]
        for (y, features, scores) in zip(y_lst, features_lst, scores_lst):
            for feature in features:
                for c in range(self.output_channel):
                    if c == y:
                        w_grad[c][feature] += scores[c] - 1
                    else:
                        w_grad[c][feature] += scores[c]
        if self.reg is not None:
            for i in range(self.output_channel):
                for j in range(self.input_channel):
                    self.w[i][j] -= 2 * self.reg * self.w[i][j]
        if self.momentum is not None:
            for i in range(self.output_channel):
                for j in range(self.input_channel):
                    self.v[i][j] = self.v[i][j] * self.momentum + self.lr * w_grad[i][j]
                    self.w[i][j] -= self.v[i][j]
        else:
            for i in range(self.output_channel):
                for j in range(self.input_channel):
                    self.w[i][j] -= self.lr * w_grad[i][j]
        return

    def train(self, features_lst, y_lst, batch_size=200):
        for i in range(0, len(features_lst), batch_size):
            features_batch = features_lst[i: i+batch_size]
            y_batch = y_lst[i: i+batch_size]
            self.backward(features_batch, y_batch)
        return 
    
    def test(self, features_lst, y_lst):
        y_pred, _ = self.forward(features_lst)
        return (
            acc(y_pred, y_lst), acc_sk(y_lst, y_pred),
            f1_micro(y_pred, y_lst), f1_sk(y_lst, y_pred, average='micro'),
            f1_macro(y_pred, y_lst), f1_sk(y_lst, y_pred, average='macro')
        )
    


