import torch, torch.nn.functional as F, numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def compute_f1(preds, y):
    
    rounded_preds = F.softmax(preds, dim=1)
    _, indices    = torch.max(rounded_preds, 1)
    correct       = (indices == y).float()
    acc           = correct.sum()/len(correct)
    y_pred        = np.array(indices.cpu().numpy())
    y_true        = np.array(y.cpu().numpy())
    result        = precision_recall_fscore_support(y_true, y_pred, average='micro')
    confusion_mat = confusion_matrix(y_true, y_pred)
    misclassifications = [(idx, i, j) for idx, (i, j) in enumerate(zip(y_true, y_pred)) if i!=j]
    return acc, result[2], result[0], result[1], confusion_mat, misclassifications, y_pred, y_true