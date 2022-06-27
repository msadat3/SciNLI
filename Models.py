import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from transformers import *
from transformers.modeling_utils import *



class BERT(nn.Module):
    def __init__(self, num_classes):
        super(BERT, self).__init__()
        self.lm = BertModel.from_pretrained("bert-base-cased", output_attentions = False, output_hidden_states = True, return_dict=False)
        self.linear = nn.Linear(768, num_classes)
    def forward(self, input, att_mask):
        _, pooled_output, _ = self.lm(input, attention_mask = att_mask)
        output = self.linear(pooled_output)
        return output

class Sci_BERT(nn.Module):
    def __init__(self, num_classes):
        super(Sci_BERT, self).__init__()
        self.lm = BertModel.from_pretrained("allenai/scibert_scivocab_cased", output_attentions = False, output_hidden_states = True, return_dict=False)
        self.linear = nn.Linear(768, num_classes)
    def forward(self, input, att_mask):
        _, pooled_output, _ = self.lm(input, attention_mask = att_mask)
        output = self.linear(pooled_output)
        return output


class XLnet(nn.Module):
    def __init__(self, num_classes):
        super(XLnet, self).__init__()
        self.lm = XLNetModel.from_pretrained("xlnet-base-cased", output_attentions = False, output_hidden_states = True,return_dict=False)
        self.sequence_summary = SequenceSummary(self.bert.config)
        self.linear = nn.Linear(768, num_classes)
    def forward(self, input, att_mask):
        output= self.lm(input, attention_mask = att_mask)
        pooled_output = self.sequence_summary(output[0])
        output = self.linear(pooled_output)
        return output


class RoBERTa(nn.Module):
    def __init__(self, num_classes):
        super(RoBERTa, self).__init__()
        self.lm = RobertaModel.from_pretrained("roberta-base", output_attentions = False, output_hidden_states = True, return_dict=False)
        self.linear = nn.Linear(768, num_classes)
    def forward(self, input, att_mask):
        _, pooled_output, _= self.lm(input, attention_mask = att_mask)
        output = self.linear(pooled_output)
        return output


