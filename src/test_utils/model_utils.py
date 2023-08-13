import torch
from transformers import AdamW
from test_utils import modeling


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