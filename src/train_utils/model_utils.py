import torch
import torch.nn as nn
from transformers import AdamW
from train_utils import modeling


def model_preds(loader, model, device, model_name):
    
    preds = []
    if model_name.startswith('bert'):
        for input_ids, seg_ids, atten_masks, label, length, task_id in loader:
            inputs = [input_ids, seg_ids, atten_masks, length]
            inputs = [e.to(device) for e in inputs]
            pred = model(inputs[0], inputs[1], inputs[2], inputs[3], task_id)
            preds.append(pred)
    else:
        for input_data, label, target, length, task_id in loader:
            inputs = [input_data, target, length]
            inputs = [e.to(device) for e in inputs]
            pred = model(inputs[0], inputs[1], inputs[2], task_id)
            preds.append(pred)
    
    return torch.cat(preds, 0)


def model_setup(config, plm_model, device):
    
    if plm_model in ['bert','bertweet']:
        model = modeling.bert_classifier(config, plm_model).to(device)
    else:
        model = modeling.lstm_classifier(config, plm_model).to(device)
    
    if not plm_model.startswith('bert'):
        optimizer = AdamW(model.parameters(), lr=float(config['lr']))
    else:
        for n, p in model.named_parameters():
            if "bert.embeddings" in n:
                p.requires_grad = False
        optimizer_grouped_parameters = [
            {'params': [p for n,p in model.named_parameters() if n.startswith('bert.encoder')], 'lr': float(config['bert_lr'])},
            {'params': [p for n,p in model.named_parameters() if n.startswith('linear')], 'lr': float(config['fc_lr'])},
            {'params': [p for n,p in model.named_parameters() if n.startswith('out')], 'lr': float(config['fc_lr'])}
            ]
        
        optimizer = AdamW(optimizer_grouped_parameters)
    
    return model, optimizer


class model_updater(object):
    
    def __init__(self, **kwargs):
        
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.plm_model = kwargs.pop("model_select")

    def update(self, trainloader, criterion, device):
        
        train_loss = []
        ind = 0
        if self.plm_model.startswith('bert'):
            for input_ids, seg_ids, atten_masks, label, length, task_id in trainloader:
                self.optimizer.zero_grad()
                inputs = [input_ids, seg_ids, atten_masks, length]
                inputs = [e.to(device) for e in inputs]
                label = label.to(device)
                outputs = self.model(inputs[0], inputs[1], inputs[2], inputs[3], task_id)
                loss = criterion(outputs, label)
                loss.backward()
                if task_id[0] == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.75)
                self.optimizer.step()
                train_loss.append(len(label) * loss.item())
        else:
            for input_data, label, target, length, task_id in trainloader:               
                self.optimizer.zero_grad()
                inputs = [input_data, target, length]
                inputs = [e.to(device) for e in inputs]
                label = label.to(device)
                outputs = self.model(inputs[0], inputs[1], inputs[2], task_id)
                loss = criterion(outputs, label)
                loss.backward()                
                if task_id[0] == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.75)
                self.optimizer.step()
                train_loss.append(len(label) * loss.item())
        
        return sum(train_loss)
        