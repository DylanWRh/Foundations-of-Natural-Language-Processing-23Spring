from datetime import datetime
import os

from torch import optim
import torch

from trainer import ParserTrainer
from parsing_model import ParsingModel
from parser_utils import load_and_preprocess_data
import json
from utils import evaluate

if __name__ == "__main__":
    # Note: Set debug to False, when training on entire corpus
    # debug = True
    debug = False

    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, train_data, dev_data, test_data, embeddings = load_and_preprocess_data(debug)
    n_features = 18 + parser.use_pos * 18 + parser.use_dep * 12
    n_classes = parser.n_trans
    parser.model = ParsingModel(embeddings, n_features=n_features, n_classes=n_classes) # You can add more arguments, depending on how you designed your parsing model

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### TODO:
    ###     1. Call an optimizer (no need to specify parameters yet, which will be implemented during training)
    ###     2. Construct the Cross Entropy Loss Function in variable `loss_func`
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam
    trainer = ParserTrainer(
        train_data=train_data,
        dev_data=dev_data,
        optimizer=optimizer,
        loss_func=loss_func,
        output_path=output_path,
        batch_size=1024,
        n_epochs=10,
        lr=0.0005,
    )
    trainer.train(parser, )

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        dependencies, relations = parser.parse(test_data)
        # with open('./prediction.json', 'w') as fh:
        #     json.dump(dependencies, fh)
        uas, las = evaluate(dependencies, relations, test_data, deprel=parser.n_deprel)  # To check the format of the input, please refer to the utils.py
        print("- test UAS: {:.2f}".format(uas * 100.0), "- test las: {:.2f}".format(las * 100.0))
        print("Done!")
