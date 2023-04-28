import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_labels = 21
        self.bert = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = 768
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, bert_input, label_ids, label_mask):
        ids_lens = bert_input[3]
        batch_size, seq_len = ids_lens.shape
        mask = ids_lens.gt(0)

        input_ids = bert_input[0]
        attention_mask = bert_input[1].type_as(mask)
        token_type_ids = bert_input[2]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_enc_out = outputs[0]

        bert_mask = attention_mask.type_as(mask)

        bert_chunks = last_enc_out[bert_mask].split(ids_lens[mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
        bert_embed = bert_out.new_zeros(batch_size, seq_len, self.hidden_size)
        bert_embed = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)
        bert_embed = self.dropout(bert_embed)

        label_predict = self.classifier(bert_embed)

        active_logits = label_predict.view(-1, self.num_labels)
        active_labels = torch.where(label_mask.view(-1), label_ids.view(-1), self.loss_fn.ignore_index)
        loss = self.loss_fn(active_logits, active_labels)

        output = label_predict.data.argmax(dim=-1)
        return loss, output

