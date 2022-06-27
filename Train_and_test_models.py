## This file contains the scripts for training and testing the pre-training language model baselines on SciNLI.

#Example command
#python Train_and_test_models.py --base '/home/msadat3/NLI/SciNLI_data/' --model_type 'XLNet' --batch_size 32 --max_length 300 --num_epochs 5 --epoch_patience 2 --device 'cuda:0' --random_seed 1234


import os
import pandas
from Train_and_test_helper import *
from Data_preparation_helper import create_data_for_pretrained_lms
import random
import torch
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Train and test pre-trained LM baselines for SciNLI.')

parser.add_argument("--base", type=str, help="Location of a directory containing the train, test and dev files in CSV format")
parser.add_argument("--model_type", type=str, help="Type of the model you want to train and test: BERT, Sci_BERT, RoBERTa or XLNet")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--max_length", type=int, default=300, help="Combined max length of sentence1 and sentence2")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train the model for.")
parser.add_argument("--epoch_patience", type=int, default=2, help="Patience for early stopping.")
parser.add_argument("--report_every", type=int, default=-1, help="Step interval to report loss. By default loss will be reported only at the end of an epoch.")
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--random_seed", type=int, default=1234)

args = parser.parse_args()

def get_numeric_label(label, label_to_idx_dict):
    if label in label_to_idx_dict.keys():
        return label_to_idx_dict[label]
    else:
        return -1

label_to_idx_dict = {
    'contrasting' : 0,
    'reasoning' : 1,
    'entailment' : 2,
    'neutral' : 3
}


base = args.base
max_length = args.max_length

epoch_patience = args.epoch_patience
model_type = args.model_type
batch_size = args.batch_size
accumulation_steps = 64//batch_size
num_epochs = args.num_epochs
num_classes = len(label_to_idx_dict.keys())
report_every = args.report_every
device = args.device


random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)


traininingSet = pandas.read_csv(base+'train.csv')
testingSet = pandas.read_csv(base+'test.csv')
validationSet = pandas.read_csv(base+'dev.csv')

traininingSet['label'] = traininingSet.apply(lambda x: get_numeric_label(x['label'], label_to_idx_dict), axis=1)    
testingSet['label'] = testingSet.apply(lambda x: get_numeric_label(x['label'], label_to_idx_dict), axis=1)    
validationSet['label'] = validationSet.apply(lambda x: get_numeric_label(x['label'], label_to_idx_dict), axis=1)    

traininingSet = traininingSet[traininingSet['label'] >= 0]
testingSet = testingSet[testingSet['label'] >= 0]
validationSet = validationSet[validationSet['label'] >= 0]

if model_type == 'Sci_BERT':
    tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_cased', do_lower_case=False)
elif model_type == 'RoBERTa' :
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=False)
elif model_type == 'XLNet':
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=False)
elif model_type == 'BERT':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
else:
    print("Unsupported model type: ", model_type)
    print("Supported model types are: 'BERT', 'Sci_BERT', 'RoBERTa', 'XLNet'")
    quit()


Prepared_data_output_location = base + '/' + model_type + '/'

create_data_for_pretrained_lms(Prepared_data_output_location, label_to_idx_dict, traininingSet, tokenizer, 'train',model_type, max_len = max_length)
create_data_for_pretrained_lms(Prepared_data_output_location, label_to_idx_dict, testingSet, tokenizer, 'test',model_type, max_len = max_length)
create_data_for_pretrained_lms(Prepared_data_output_location, label_to_idx_dict, validationSet, tokenizer, 'valid',model_type, max_len = max_length)


X_train = load_data(Prepared_data_output_location+'X_train.pkl')
X_test = load_data(Prepared_data_output_location+'X_test.pkl')
X_valid = load_data(Prepared_data_output_location+'X_valid.pkl')

att_mask_train = load_data(Prepared_data_output_location+'att_mask_train.pkl')
att_mask_test = load_data(Prepared_data_output_location+'att_mask_test.pkl')
att_mask_valid = load_data(Prepared_data_output_location+'att_mask_valid.pkl')
  
y_train = load_data(Prepared_data_output_location+'y_train.pkl')
y_test = load_data(Prepared_data_output_location+'y_test.pkl')
y_valid = load_data(Prepared_data_output_location+'y_valid.pkl')


y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_valid = np.asarray(y_valid)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_valid = np.asarray(X_valid)


model_location = Prepared_data_output_location+'model/'
if p.exists(model_location) == False:
    os.mkdir(model_location)


train_model(model_location, model_type, X_train, att_mask_train, y_train, X_valid, att_mask_valid, y_valid, device,batch_size,accumulation_steps,num_epochs,num_classes,report_every, epoch_patience)
test_model(model_location, model_type, X_test, att_mask_test, y_test, device, batch_size,num_classes)
