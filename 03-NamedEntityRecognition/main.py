import torch
from model import Model
from transformers import BertModel, BertTokenizer
from data import batch_variable, load_data, DataLoader


raw_labels = ['NHCS', 'NHVI', 'NCSM', 'NCGV', 'NASI', 'NT', 'NS', 'NO', 'NATS', 'NCSP']
true_labels = ('O', 'B-NHCS', 'I-NHCS', 'B-NHVI', 'I-NHVI', 'B-NCSM', 'I-NCSM', 'B-NCGV', 'I-NCGV', 'B-NASI', 'I-NASI', 'B-NT', 'I-NT', 'B-NS', 'I-NS', 'B-NO', 'I-NO', 'B-NATS', 'I-NATS', 'B-NCSP', 'I-NCSP')
label2id = {k:v for v,k in enumerate(true_labels)}
id2label = {v:k for k,v in label2id.items()}
rawlabel2id = {k:v for v,k in enumerate(raw_labels)}


def inference(model, test_loader, tokenizer):
    model.eval()
    pred_all = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            bert_input, label_ids, label_mask = batch_variable(batch_data, tokenizer)
            _, preds = model(bert_input, label_ids, label_mask)

            for i, sentence_mask in enumerate(label_mask):
                for j, word_mask in enumerate(sentence_mask):
                    if word_mask.item() == False:
                        preds[i][j] = 0

            pred_lst = []
            for pred in preds:
                pred_lst.append([id2label[pre.cpu().item()] for pre in pred])
            
            pred_all += pred_lst
    return pred_all


def save_json(path, data, preds):
    with open(path, 'w', encoding='utf-8') as f:
        for line, pred in zip(data, preds):
            res = {'id': line['id']}
            true_pred = pred[:len(line['context'])] + ['O']
            entity = [{'label': label, 'span': []} for label in raw_labels]
            
            i = 0
            while i < len(true_pred):
                label = true_pred[i]
                if label == 'O':
                    i += 1
                    continue
                if label[1] == '-':
                    start = i
                    while i < len(true_pred)-1 and true_pred[i+1] != 'O' and true_pred[i+1][2:] == true_pred[i][2:]:
                        i += 1
                    end = i+1
                    entity[rawlabel2id[label[2:]]]['span'].append(f'{start};{end}')
                i += 1
            res['entities'] = entity
            f.write(str(res).replace("'", '"')+'\n')                

            
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    weight = torch.load('model.ckpl', map_location=device)
    model.load_state_dict(weight, strict=False)
    
    batch_size = 16
    
    test_path = './input/test_data.json'
    data = load_data(test_path)
    test_loader = DataLoader(data, batch_size)
    
    tokenizer = BertTokenizer.from_pretrained('tokenizer')
    pred = inference(model, test_loader, tokenizer)
    
    save_path = './output/output.json'
    save_json(save_path, data, pred)


if __name__ == '__main__':
    main()