import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


# Evaluation metrics
def train_compute_f1(preds, y, aux_eval=False):
    
    rounded_preds = F.softmax(preds, dim=1)
    _, indices = torch.max(rounded_preds, dim=1)
    
    y_pred = np.array(indices.cpu().numpy())
    y_true = np.array(y.cpu().numpy())
    
    if not aux_eval:
        result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2])
        f1_average = (result[2][0] + result[2][2]) / 2 # average F1 of FAVOR and AGAINST labels
    else:
        result = precision_recall_fscore_support(y_true, y_pred, average='micro')
        f1_average = result[2]
        
    return f1_average