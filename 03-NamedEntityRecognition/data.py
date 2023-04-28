import json 
import torch


raw_labels = ['NHCS', 'NHVI', 'NCSM', 'NCGV', 'NASI', 'NT', 'NS', 'NO', 'NATS', 'NCSP']
true_labels = ('O', 'B-NHCS', 'I-NHCS', 'B-NHVI', 'I-NHVI', 'B-NCSM', 'I-NCSM', 'B-NCGV', 'I-NCGV', 'B-NASI', 'I-NASI', 'B-NT', 'I-NT', 'B-NS', 'I-NS', 'B-NO', 'I-NO', 'B-NATS', 'I-NATS', 'B-NCSP', 'I-NCSP')
label2id = {k:v for v,k in enumerate(true_labels)}
id2label = {v:k for k,v in label2id.items()}


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


class DataLoader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
    
    def __iter__(self):
        batch = []
        for idx in range(len(self.data)):
            batch.append(self.data[idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch):
            yield batch
    
    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


def batch_variable(batch_data, tokenizer):
    sentence_lst = []
    labels_lst = []
    for idx, item in enumerate(batch_data):
        sentence = item['context']
        labels = ['O'] * len(sentence)

        for e in item['entities']:
            label = e['label']
            span = e['span']
            for s in span:
                start, end = [int(i) for i in s.split(';')]
                if start == end-1:
                    labels[start] = 'B-' + label
                else:
                    labels[start] = 'B-' + label
                    labels[start+1:end] = ['I-' + label] * (end-start-1)
        sentence_lst.append(list(sentence))
        labels_lst.append(labels)
    
    batch_size = len(batch_data)
    max_seq_len = max([len(item['context']) for item in batch_data])
    label_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    for idx, labels in enumerate(labels_lst):
        label_ids[idx, :len(labels)] = torch.tensor([label2id[label] for label in labels])
        label_mask[idx, :len(labels)].fill_(1)
    
    ids_lst, lens_lst, segment_lst, mask_lst = word2id(sentence_lst, tokenizer)
    ids_lst_ = torch.LongTensor(ids_lst)
    lens_lst_ = torch.LongTensor(lens_lst)
    segment_lst_ = torch.LongTensor(segment_lst)
    mask_lst_ = torch.LongTensor(mask_lst)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_input = [ids_lst_.to(device), mask_lst_.to(device), segment_lst_.to(device), lens_lst_.to(device)]

    return bert_input, label_ids.to(device), label_mask.to(device)


def word2id(sentence_lst, tokenizer):
    ids_lst = []
    lens_lst = []
    segment_lst = []
    mask_lst = []

    max_ids_len = 0
    max_lens_len = 0

    for sentence in sentence_lst:
        ids = []
        lens = []
        for word in sentence:
            word_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            for i in word_id:
                ids.append(i)
            lens.append(len(word_id))
        max_ids_len = max(max_ids_len, len(ids))
        max_lens_len = max(max_lens_len, len(lens))

        ids_lst.append(ids)
        lens_lst.append(lens)

    mask_lst = [[1] * len(ids) for ids in ids_lst]
    for idx, item in enumerate(ids_lst):
        pad_size = max_ids_len - len(item)
        ids_lst[idx] += [0] * pad_size
        mask_lst[idx] += [0] * pad_size
    
    for idx, item in enumerate(lens_lst):
        pad_size = max_lens_len - len(item)
        lens_lst[idx] += [0] * pad_size
    
    segment_lst = [[0] * len(ids) for ids in ids_lst]
    return ids_lst, lens_lst, segment_lst, mask_lst

