from transformers import *
import json
from Utils import *
import os.path as p
import os
import pandas
import numpy as np
import random
from transformers import RobertaTokenizer

long_count = 0

def Tokenize_Input(first_sentence, second_sentence, tokenizer, model_type):

    first_encoded = tokenizer.encode(str(first_sentence),add_special_tokens=False)
    second_encoded = tokenizer.encode(str(second_sentence), add_special_tokens=False)

    if model_type!= 'XLNet':
        encoded = [tokenizer.cls_token_id] +first_encoded + [tokenizer.sep_token_id] + second_encoded + [tokenizer.sep_token_id] 
    else:
        encoded = first_encoded + [tokenizer.sep_token_id] + second_encoded + [tokenizer.sep_token_id] + [tokenizer.cls_token_id] 
    return encoded

def get_attention_masks(X, tokenizer):
    #copied and then updated from: https://github.com/bino282/bert4news/blob/master/train_pytorch.py
    attention_masks = []

    # For each sentence...
    for sent in X:
        att_mask = [int(token_id != tokenizer.pad_token_id) for token_id in sent]

        # Store the attention mask for this sentence.
        att_mask = np.asarray(att_mask)
        attention_masks.append(att_mask)
    attention_masks = np.asarray(attention_masks)
    return attention_masks

def pad_seq(seq,max_len,pad_idx):
    if len(seq)>max_len:
        sep = seq[-1]
        seq = seq[0:max_len-1]
        seq.append(sep)
    while len(seq) != max_len:
        seq.append(pad_idx)
    return seq

def create_data_for_pretrained_lms(output_location, label_to_idx_dict, data_subset, tokenizer, suffix, model_type, max_len=300):
    if p.exists(output_location) == False:
        os.mkdir(output_location)

    X = data_subset.apply(lambda x: Tokenize_Input(x['sentence1'], x['sentence2'], tokenizer, model_type), axis=1)    
    X = pandas.Series(X)


    actual_max_len = 0
    for x in X:
        if len(x) > actual_max_len:
            actual_max_len = len(x)
    max_len = min(max_len,actual_max_len)

    X = X.apply(pad_seq, max_len=max_len, pad_idx=tokenizer.pad_token_id)
    X = np.array(X.values.tolist())
    att_mask = get_attention_masks(X, tokenizer)

    save_data(X, output_location + 'X_'+suffix+'.pkl')
    save_data(att_mask, output_location + 'att_mask_'+suffix+'.pkl')
    
    print(suffix+' sample count and length')
    print(X.shape)
    
    y = np.array(data_subset['label'].tolist())
    save_data(y, output_location + 'y_'+suffix+'.pkl')
    


