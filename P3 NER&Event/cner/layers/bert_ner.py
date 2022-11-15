# by wucx, 2022/10/20
# bert+softmax, bert+crf, bert+span

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from torch.nn import CrossEntropyLoss
from layers.crf import CRF


class BertSoftmaxNER(nn.Module):
    def __init__(self, conf):
        super(BertSoftmaxNER, self).__init__()
        self.num_classes = conf.num_classes
        self.bert = BertModel.from_pretrained(conf.bert_dir)
        self.bert_conf = BertConfig.from_pretrained(conf.bert_dir)

        self.drop = nn.Dropout(conf.dropout)
        self.classifier = nn.Linear(self.bert_conf.hidden_size, conf.num_classes)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        seq_output = outputs[0]
        seq_output = self.drop(seq_output)
        logits = self.classifier(seq_output)
        # Append hidden states and attention, which may be useful
        outputs = (logits, ) + outputs[2:]  

        # Calculate loss for training
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_classes)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits,active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            outputs = (loss, ) + outputs
        return outputs # (loss), logits, (hidden_states), (attentions)



class BertCrfNER(nn.Module):
    def __init__(self, conf):
        super(BertCrfNER, self).__init__()
        self.num_classes = conf.num_classes
        self.bert = BertModel.from_pretrained(conf.bert_dir)
        self.bert_conf = BertConfig.from_pretrained(conf.bert_dir)

        self.drop = nn.Dropout(conf.dropout)
        self.classifier = nn.Linear(self.bert_conf.hidden_size, conf.num_classes)
        self.crf = CRF(num_tags=conf.num_classes, batch_first=True)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        seq_output = outputs[0]
        seq_output = self.drop(seq_output)
        logits = self.classifier(seq_output)
        outputs = (logits, )

        # Calculate loss for training
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss, ) + outputs
        return outputs # (loss), scores



class StartFFN(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states):
        x = self.linear(hidden_states)
        return x


class EndFFN(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions):
        x = self.linear1(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.tanh(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x
        

class BertSpanNER(nn.Module):
    def __init__(self, conf):
        super(BertSpanNER, self).__init__()
        # self.soft_label = True
        self.num_classes = conf.num_classes
        self.bert = BertModel.from_pretrained(conf.bert_dir)
        self.bert_conf = BertConfig.from_pretrained(conf.bert_dir)

        self.drop = nn.Dropout(conf.dropout)
        self.start_fc = StartFFN(self.bert_conf.hidden_size, self.num_classes)
        self.end_fc = EndFFN(self.bert_conf.hidden_size + self.num_classes, self.num_classes)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_labels=None, end_labels=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        seq_output = outputs[0]
        seq_output = self.drop(seq_output)
        start_logits = self.start_fc(seq_output)

        # during training, use the true start labels for the predictions of the end labels
        # during evaluation, the predicted start labels (true start labels are not availabel) are used.
        if self.training: 
            # label_logits = torch.FloatTensor(input_ids.size(0), input_ids.size(1), self.num_classes)
            # label_logits.zero_()
            label_logits = torch.zeros((input_ids.size(0), input_ids.size(1), self.num_classes), dtype=torch.float)
            label_logits = label_logits.to(input_ids.device)
            label_logits.scatter_(2, start_labels.unsqueeze(2), 1)
        else:
            label_logits = F.softmax(start_logits, -1)

        end_logits = self.end_fc(seq_output, label_logits)
        outputs = (start_logits, end_logits, ) + outputs[2:]

        # Calculate loss for training
        if start_labels is not None and end_labels is not None:
            loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_classes)
            end_logits = end_logits.view(-1, self.num_classes)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]
            active_start_labels = start_labels.view(-1)[active_loss]
            active_end_labels = end_labels.view(-1)[active_loss]            

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss, ) + outputs
        return outputs
