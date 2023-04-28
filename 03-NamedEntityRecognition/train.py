import torch
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from data import batch_variable, load_data, DataLoader
from seqeval.metrics import accuracy_score, classification_report, f1_score
from transformers import BertModel, BertTokenizer
from model import Model
from tqdm import tqdm
import random


def train(model, train_loader, eval_loader):
    best_f1 = float('-inf')
    avg_loss = []
    optimizer = optim.Adam(params=model.parameters(), lr=1e-5)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    n_epoches = 20
    for epoch in range(n_epoches):
        train_true, train_tot = 0, 0
        for batch_idx, batch_data in tqdm(enumerate(train_loader)):
            model.train()
            bert_input, label_ids, label_mask = batch_variable(batch_data, tokenizer)
            loss, preds = model(bert_input, label_ids, label_mask)

            avg_loss.append(loss.data.item())

            batch_true = ((preds == label_ids) * label_mask).sum().item()
            batch_tot = label_mask.sum().item()

            train_true += batch_true
            train_tot += batch_tot

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 60 == 0:
                print(f'Epoch: {epoch+1}, Iter: {batch_idx+1}, train_loss: {np.array(avg_loss).mean()}, train_acc: {train_true / train_tot}')
        
        scheduler.step()

        eval_loss, eval_acc, eval_f1, eval_report = evaluate(model, eval_loader, tokenizer)
        print(f'eval_loss: {eval_loss}, eval_acc: {eval_acc}, eval_f1: {eval_f1}')
        print(eval_report)

        if best_f1 < eval_f1:
            best_f1 = eval_f1
            torch.save(model.state_dict(), 'model.ckpl')
            tokenizer.save_pretrained('tokenizer')

def evaluate(model, eval_loader, tokenizer):

    true_labels = ('O', 'B-NHCS', 'I-NHCS', 'B-NHVI', 'I-NHVI', 'B-NCSM', 'I-NCSM', 'B-NCGV', 'I-NCGV', 'B-NASI', 'I-NASI', 'B-NT', 'I-NT', 'B-NS', 'I-NS', 'B-NO', 'I-NO', 'B-NATS', 'I-NATS', 'B-NCSP', 'I-NCSP')
    label2id = {k:v for v,k in enumerate(true_labels)}
    id2label = {v:k for k,v in label2id.items()}

    model.eval()
    loss_tot = 0
    pred_all = []
    label_all = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_loader):
            bert_input, label_ids, label_mask = batch_variable(batch_data, tokenizer)
            loss, preds = model(bert_input, label_ids, label_mask)

            loss_tot += loss
            for i, sentence_mask in enumerate(label_mask):
                for j, word_mask in enumerate(sentence_mask):
                    if word_mask.item() == False:
                        preds[i][j] = 0
            labels_lst = []
            for ids in label_ids:
                labels_lst.append([id2label[i.cpu().item()] for i in ids])
            pred_lst = []
            for pred in preds:
                pred_lst.append([id2label[pre.cpu().item()] for pre in pred])
            
            label_all += labels_lst
            pred_all += pred_lst
    
    acc = accuracy_score(label_all, pred_all)
    f1 = f1_score(label_all, pred_all, average='micro')
    report = classification_report(label_all, pred_all, digits=3, output_dict=False)

    return loss_tot / len(eval_loader), acc, f1, report


def main():
    train_path = 'train_data.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16 
    model = Model().to(device)
    
    data = load_data(train_path)
    random.seed(42)
    train_N = int(0.8 * len(data))
    random.shuffle(data)
    
    train_data = data[:train_N]
    eval_data = data[train_N:]
    
    train_loader = DataLoader(train_data, batch_size)
    eval_loader = DataLoader(eval_data, batch_size)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    param = count_parameters(model)
    print(param/1000000)
    
    train(model, train_loader, eval_loader)

if __name__ == '__main__':
    main()