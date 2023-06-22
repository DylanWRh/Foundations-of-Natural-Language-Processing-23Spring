from datetime import datetime
import os
import torch
import numpy as np
from parsing_model import ParsingModel
from parser_utils import load_and_preprocess_data
import json
from utils import evaluate

if __name__ == '__main__':

    parser, train_data, dev_data, test_data, embeddings = load_and_preprocess_data(False)

    n_features = 18 + parser.use_pos * 18 + parser.use_dep * 12
    n_classes = parser.n_trans
    parser.model = ParsingModel(embeddings, n_features=n_features, n_classes=n_classes) 
    model_path = f'./results/model.weights'
    parser.model.load_state_dict(torch.load(model_path))
    parser.model.eval()
    dependencies, relations = parser.parse(test_data)

    json_res = [None] * len(dependencies)
    for i, (dependency, relation) in enumerate(zip(dependencies, relations)):
        dep_res = [-1] * (len(dependency) + 1)
        rel_res = [-1] * (len(relation) + 1)
        for (h, t), r in zip(dependency, relation):
            dep_res[t] = h
            rel_res[t] = parser.id2tran[r][2:]
        json_res[i] = list(zip(dep_res[1:], rel_res[1:]))
    with open('./prediction.json', 'w') as fh:
        json.dump(json_res, fh)

    uas, las = evaluate(dependencies, relations, test_data, deprel=parser.n_deprel) 
    print("- test UAS: {:.2f}".format(uas * 100.0), "- test las: {:.2f}".format(las * 100.0))