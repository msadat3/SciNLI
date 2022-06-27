from Models import *
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
import numpy as np
import torch.optim as optim
from Utils import *
import os.path as p
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import random
import numpy as np
import pandas
from torch.utils.data.sampler import Sampler
import math


def train_model(location, model_type, X_train, att_mask_train, y_train, X_valid, att_mask_valid, y_valid, device, batch_size,accumulation_steps,num_epochs,num_classes,report_every, epoch_patience):
    x_tr = torch.tensor(X_train, dtype=torch.long)
    att_mask_tr = torch.tensor(att_mask_train, dtype=torch.long)
    y_tr = torch.tensor(y_train, dtype=torch.long)

    x_val = torch.tensor(X_valid, dtype=torch.long)
    att_mask_val = torch.tensor(att_mask_valid, dtype=torch.long)
    y_val = torch.tensor(y_valid, dtype=torch.long)

    train = TensorDataset(x_tr, y_tr, att_mask_tr)
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val = TensorDataset(x_val, y_val, att_mask_val)
    validationLoader = DataLoader(val, batch_size=64)
    
    if model_type == "Sci_BERT":
        model = Sci_BERT(num_classes)
    elif model_type == "RoBERTa":
        model = RoBERTa(num_classes)
    elif model_type == "XLNet":
        model = XLnet(num_classes)
    else:
        model = BERT(num_classes)

    if device != 'cpu':
        model.to(torch.device(device))
    
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    prev_best_score = -1
    notImprovingEpoch = 0

    for epoch in range(0,num_epochs):
        if notImprovingEpoch == epoch_patience:
            print('Performance not improving for '+str(epoch_patience) +' consecutive epochs. Stopping training.')
            break
        model.train()

        i = 0
        step_count = 0
        optimizer.zero_grad()
        for data, target, att in trainloader:
            data = data.to(torch.device(device))
            target = target.to(torch.device(device))
            att = att.to(torch.device(device))
            output = model(data, att)
            loss = criterion(output, target)/accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i+1) == len(trainloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                step_count+=1
                if step_count % report_every == 0 and report_every!=-1:
                    print('Epoch', epoch, 'step', step_count, "loss", loss.item(), 'current val f1', prev_best_score)
            i += 1
        model.eval()
        n = 0
        print("=============Epoch " + str(epoch) + " =============")
        with torch.no_grad():
            val_out = []
            for val_data, val_target, att in validationLoader:
                val_data = val_data.to(torch.device(device))
                val_target = val_target.to(torch.device(device))
                att = att.to(torch.device(device))

                out = model(val_data, att)
                out = torch.argmax(out, dim=1)

                out = out.cpu().detach().numpy()
                val_out += out.tolist()
                n += len(val_target)
            current_score = f1_score(y_valid.tolist(), val_out, average="macro")
            if current_score > prev_best_score:
                print("Validation f1 score improved from", prev_best_score, "to", current_score, "saving model...")
                prev_best_score = current_score
                last_checkpoint_info = {
                    'epoch': epoch,
                    'score' : prev_best_score
                }
                save_data(last_checkpoint_info,location+'/checkpoint.pkl')
                torch.save(model.state_dict(), location + '/model.pt')
                notImprovingEpoch = 0
            else:
                print("Validation f1 score did not improve from", prev_best_score)
                notImprovingEpoch += 1

def test_model(location, model_type, x, att_x, y, device, batch_size, num_classes):
    if model_type == "Sci_BERT":
        test_model = Sci_BERT(num_classes)
    elif model_type == "RoBERTa":
        test_model = RoBERTa(num_classes)
    elif model_type == "XLNet":
        test_model = XLnet(num_classes)
    else:
        test_model = BERT(num_classes)

    if device != 'cpu':
        test_model.to(torch.device(device))

    test_model.load_state_dict(torch.load(location + '/model.pt', map_location='cuda:0'))

    x_te = torch.tensor(x, dtype=torch.long)
    att_mask_te = torch.tensor(att_x, dtype=torch.long)
    y_te = torch.tensor(y, dtype=torch.long)
    te = TensorDataset(x_te, y_te, att_mask_te)
    testLoader = DataLoader(te, batch_size=batch_size)

    test_model.eval()

    with torch.no_grad():
        test_out = []
        #all_probabilities = []
        #all_max_probabilities = []
        for test_data, test_target, att in testLoader:
            test_data = test_data.to(torch.device(device))
            test_target = test_target.to(torch.device(device))
            att = att.to(torch.device(device))

            out = test_model(test_data, att)
            #probabilities = torch.softmax(out, dim=1)
            out = torch.argmax(out, dim=1)
            out = out.cpu().detach().numpy()
            test_out += out.tolist()
            #probabilities = probabilities.cpu().detach().numpy().tolist()
            #max_probabilities = [max(p) for p in probabilities]

            #all_max_probabilities+=max_probabilities
            #all_probabilities+=probabilities

        #save_data(test_out,location+'test_output.pkl')
        #save_data(all_probabilities,location+'all_probabilities.pkl')
        current_score = f1_score(y.tolist(), test_out, average="macro")
        Accuracy = accuracy_score(y.tolist(), test_out)

        print('Macro f1', current_score)
        print('Accuracy', Accuracy)

        print(classification_report(y.tolist(), test_out, target_names=['contrasting', 'reasoning', 'entailment', 'neutral'], digits=4))
        


