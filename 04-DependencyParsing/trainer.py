import torch
import torch.nn as nn
from utils import evaluate
import math
from tqdm import tqdm 
import numpy as np

class ParserTrainer():

    def __init__(
        self,
        train_data,
        dev_data,
        optimizer,
        loss_func,
        output_path,
        batch_size=1024,
        n_epochs=10,
        lr=0.0005, 
    ): # You can add more arguments
        """
        Initialize the trainer.
        
        Inputs:
            - train_data: Packed train data
            - dev_data: Packed dev data
            - optimizer: The optimizer used to optimize the parsing model
            - loss_func: The cross entropy function to calculate loss, initialized beforehand
            - output_path (str): Path to which model weights and results are written
            - batch_size (int): Number of examples in a single batch
            - n_epochs (int): Number of training epochs
            - lr (float): Learning rate
        """
        self.train_data = train_data
        self.dev_data = dev_data
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.output_path = output_path
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        ### TODO: You can add more initializations here


    def train(self, parser, ): # You can add more arguments as you need
        """
        Given packed train_data, train the neural dependency parser (including optimization),
        save checkpoints, print loss, log the best epoch, and run tests on packed dev_data.

        Inputs:
            - parser (Parser): Neural Dependency Parser
        """
        best_dev_UAS = best_dev_LAS = 0

        ### TODO: Initialize `self.optimizer`, i.e., specify parameters to optimize
        self.optimizer = self.optimizer(parser.model.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.n_epochs))
            dev_UAS, dev_LAS = self._train_for_epoch(parser, )
            # TODO: you can change this part, to use either uas or las to select best model
            if parser.unlabeled:
                if dev_UAS > best_dev_UAS:
                    best_dev_UAS = dev_UAS
                    print("New best dev UAS! Saving model.")
                    torch.save(parser.model.state_dict(), self.output_path)
            else:
                if dev_LAS > best_dev_LAS:
                    best_dev_LAS = dev_LAS
                    print("New best dev LAS! Saving model.")
                    torch.save(parser.model.state_dict(), self.output_path)
            print("")


    def _train_for_epoch(self, parser, ): # You can add more arguments as you need
        """ 
        Train the neural dependency parser for single epoch.

        Inputs:
            - parser (Parser): Neural Dependency Parser
        Return:
            - dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
        """
        parser.model.train() # Places model in "train" mode, e.g., apply dropout layer, etc.
        ### TODO: Train all batches of train_data in an epoch.
        ### Remember to shuffle before training the first batch (You can use Dataloader of PyTorch)

        n_batch = math.ceil(len(self.train_data) / self.batch_size)
        with tqdm(total=n_batch) as prog:
            for i, (train_x, train_y) in enumerate(get_batch(self.train_data, self.batch_size)):
                self.optimizer.zero_grad()
                train_x = torch.from_numpy(train_x).long()
                train_y = torch.from_numpy(train_y).long()
                logits = parser.model(train_x)
                loss = self.loss_func(logits, train_y)
                loss.backward()
                self.optimizer.step()

                prog.update(1)

        print("Evaluating on dev set",)
        parser.model.eval() # Places model in "eval" mode, e.g., don't apply dropout layer, etc.
        dependencies, relations = parser.parse(self.dev_data)
        
        gold_dependencies = self.dev_data
        uas, las = evaluate(dependencies, relations, gold_dependencies, deprel=parser.n_deprel)  # To check the format of the input, please refer to the utils.py
        if parser.unlabeled:
            las = 0
        print("- dev UAS: {:.2f}".format(uas * 100.0), "- dev LAS: {:.2f}".format(las * 100.0))
        return uas, las


def get_batch(data, batch_size):
    X = np.array([d[0] for d in data])
    y = np.array([d[2] for d in data])
    return get_batch_(X, y, batch_size)


def get_batch_(X, y, batch_size):
    data_len = len(X)
    idx = np.arange(data_len)
    np.random.shuffle(idx)
    for i in range(0, data_len, batch_size):
        batch_idx = idx[i:i+batch_size]
        yield X[batch_idx], y[batch_idx]