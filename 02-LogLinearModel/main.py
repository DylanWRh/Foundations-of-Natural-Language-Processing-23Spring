from utils import get_features, _shuffle
from model import LogLinear
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate',      type=float, default=0.001)
parser.add_argument('--output_channel',     type=int,   default=20)
parser.add_argument('--input_channel',      type=int,   default=20000)
parser.add_argument('--batch_size',         type=int,   default=200)
parser.add_argument('--n_epochs',           type=int,   default=5)
parser.add_argument('--has_reg',            type=bool,  default=True)
parser.add_argument('--regularization',     type=float, default=1e-6)
parser.add_argument('--has_momentum',       type=bool,  default=True)
parser.add_argument('--momentum',           type=float, default=0.9)
parser.add_argument('--seed',               type=int,   default=42)
args = parser.parse_args()

if not args.has_reg:
    args.regularization = None
if not args.has_momentum:
    args.momentum = None

train_file = 'train.csv'
test_file = 'test.csv'

train_features, train_labels, word_embed = get_features(train_file, dim=args.input_channel)
test_features, test_labels = get_features(test_file, dim=args.input_channel, word_embed=word_embed)
model = LogLinear(
    lr=args.learning_rate, 
    output_channel=args.output_channel, 
    input_channel=args.input_channel, 
    reg=args.regularization, 
    momentum=args.momentum, 
    seed=args.seed
)

best_acc = 0
best_micro = 0
best_macro = 0
best_acc_model = None
best_micro_model = None 
best_macro_model = None
best_acc_epoch = 0
best_micro_epoch = 0
best_macro_epoch = 0

# with open('result.txt', 'w') as f:
for i in range(args.n_epochs):
    train_features, train_labels = _shuffle(train_features, train_labels)
    model.train(train_features, train_labels)
    train_acc, train_acc_sk, train_f1_micro, train_f1_micro_sk, train_f1_macro, train_f1_macro_sk = model.test(train_features, train_labels)
    test_acc, test_acc_sk, test_f1_micro, test_f1_micro_sk, test_f1_macro, test_f1_macro_sk = model.test(test_features, test_labels)
    if test_acc > best_acc:
            best_acc = test_acc
            best_acc_model = model.w
            best_acc_epoch = i+1
    if test_f1_micro > best_micro:
        best_micro = test_f1_micro
        best_micro_model = model.w
        best_micro_epoch = i+1
    if test_f1_macro > best_macro:
        best_macro = test_f1_macro
        best_macro_model = model.w
        best_macro_epoch = i+1
    print(
        f'''    Epoch {i+1}, 
        Training Acc = {train_acc} | {train_acc_sk}
        Training Micro = {train_f1_micro} | {train_f1_micro_sk}
        Training Macro = {train_f1_macro} | {train_f1_macro_sk}
        Testing Acc = {test_acc} | {test_acc_sk}
        Testing Micro = {test_f1_micro} | {test_f1_micro_sk}
        Testing Macro = {test_f1_macro} | {test_f1_macro_sk}
        best_acc = {best_acc}, epoch = {best_acc_epoch}
        best_micro = {best_micro}, epoch = {best_micro_epoch}
        best_macro = {best_macro}, epoch = {best_macro_epoch}'''
    )
    # f.write(
    #     f'''    Epoch {i+1}, 
    #     Training Acc = {train_acc} | {train_acc_sk}
    #     Training Micro = {train_f1_micro} | {train_f1_micro_sk}
    #     Training Macro = {train_f1_macro} | {train_f1_macro_sk}
    #     Testing Acc = {test_acc} | {test_acc_sk}
    #     Testing Micro = {test_f1_micro} | {test_f1_micro_sk}
    #     Testing Macro = {test_f1_macro} | {test_f1_macro_sk}
    #     best_acc = {best_acc}, epoch = {best_acc_epoch}
    #     best_micro = {best_micro}, epoch = {best_micro_epoch}
    #     best_macro = {best_macro}, epoch = {best_macro_epoch}\n'''
    # )