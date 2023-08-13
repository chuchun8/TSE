import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


# Evaluation metrics
def eval_compute_f1(preds, y, gt_tar, mapped_tar):
    
    rounded_preds = F.softmax(preds, dim=1)
    _, indices = torch.max(rounded_preds, dim=1)
    
    # Precision, recall and F1 metrics for TSE task
    count = 0
    correct = (indices == y).float()
    for i in range(len(gt_tar)):
        if gt_tar[i] == mapped_tar[i] and correct[i] == 1.:
            count += 1
    count_x = len([x for x in mapped_tar if x != ['unrelated']])
    count_y = len([x for x in gt_tar if x != ['unrelated']])
    precision = count / count_x
    recall = count / count_y
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Accuracy for TSE task
    count_acc = 0
    for i in range(len(gt_tar)):
        if gt_tar[i] == ["unrelated"] and mapped_tar[i] == ["unrelated"]:
            count_acc += 1
        elif gt_tar[i] == mapped_tar[i] and correct[i] == 1.:
            count_acc += 1  
    acc = count_acc / len(correct)
        
    return [precision, recall, f1, acc]
