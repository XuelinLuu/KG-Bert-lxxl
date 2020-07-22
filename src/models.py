import torch.nn as nn
import transformers

class BertKG(nn.Module):
    def __init__(self, bert_path, num_labels):
        super(BertKG, self).__init__()
        self.bert_path = bert_path
        self.num_labels = num_labels
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = output[1]

        logist = self.dropout(pooled_output)
        return self.classifier(logist)