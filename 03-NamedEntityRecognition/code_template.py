import argparse
import torch.nn as nn


def load_data(train_data_path, test_data_path):
    """
    Load, split and pack your data.
    Input: The paths to the files of your original data
    Output: Packed data, e.g., a list like [train_data, dev_data, test_data]
    """
    pass


class Model(nn.Module):
    def __init__(self, ): # You can add more arguments
        pass
    def forward(): # You can add more arguments
        """
        The implementation of NN forward function.
        Input: The data of your batch_size
        Output: The result tensor of this batch
        """
        pass


class Trainer():
    def __init__(self, model: Model, ): # You can add more arguments
        self.model = model
        pass
    def train(train_data, dev_data, ): # You can add more arguments
        """
        Given packed train_data, train the model (including optimization),
        save checkpoints, print loss, log the best epoch, and run tests on packed dev_data
        """
        pass
    def test(data, mode, ): # You can add more arguments
        """
        Given packed data, run the model and predict results
        This function should be able to load a model from a checkpoint

        """
        if mode == 'dev_eval':
            pass # Directly run tests on dev_data and print results in the console
        elif mode == 'test_eval':
            pass # Here you should save the results to ./output/output.json
        else:
            pass


def main(args):
    # NOTE: You can use variables in args as further arguments of the following functions
    train_data_path = './input/train_data.json'
    test_data_path = './input/test_data.json'
    train_data, dev_data, test_data = load_data(train_data_path, test_data_path)
    model = Model()
    trainer = Trainer(model, )
    trainer.train(train_data, dev_data, )
    trainer.test(test_data, mode='test_eval')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="arguments")
    # You can add more arguments as you want
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch Size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Epochs"
    )
    args = parser.parse_args()
    main(args)