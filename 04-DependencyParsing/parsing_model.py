import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ParsingModel(nn.Module):

    def __init__(self, embeddings, n_features=36, hidden_size=256, n_classes=3, dropout=0.5):
        """ 
        Initialize the parser model. You can add arguments/settings as you want, depending on how you design your model.
        NOTE: You can load some pretrained embeddings here (If you are using any).
              Of course, if you are not planning to use pretrained embeddings, you don't need to do this.
        """
        super(ParsingModel, self).__init__()
        pass
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.embedding_size = embeddings.shape[1]
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embeddings.astype(np.float32)))

        self.f2h = nn.Linear(self.n_features * self.embedding_size, self.hidden_size)
        self.h2l = nn.Linear(self.hidden_size, self.n_classes)
        self.dropout = nn.Dropout(self.dropout)
        self.activation = nn.ReLU()

    def forward(self, t):
        """
        Input: input tensor of tokens -> SHAPE (batch_size, n_features)
        Return: tensor of predictions (output after applying the layers of the network
                                 without applying softmax) -> SHAPE (batch_size, n_classes)
        """
        
        bs = t.shape[0]
        t = self.embedding.weight[t].view(bs, -1)
        t = self.f2h(t)
        t = self.activation(t)
        if self.training:
            t = self.dropout(t)
        logits = self.h2l(t)
        
        return logits
