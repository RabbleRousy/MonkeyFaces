import torch
from torcheval.metrics.functional import multiclass_f1_score

def compute_confusion_matrix(class_num_count, preds, labels, n_category:int):
    """
    Calculate the metrics for each category.
    """
    # count the number of TP, FP and FN for each category
    for category in range(n_category):
        n_TP = torch.count_nonzero(preds==labels)
        n_FP = max(0, torch.count_nonzero(labels==category)-n_TP)
        n_FN = max(0, torch.count_nonzero(preds==category) -n_TP)
        class_num_count[category][0] += n_TP
        class_num_count[category][1] += n_FP
        class_num_count[category][2] += n_FN

    return class_num_count

def compute_precision_recall_f1(TP, FP, FN):
    """
    Calculate f1-score for each class.
    """
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    F1 = 2*precision*recall / (precision+recall)
    return [precision, recall, F1]

def compute_multiclass_f1_score(input, target, num_classes, average):
    """
    Use the function from site-package to calculate f1-score for multi-class.
    """
    return multiclass_f1_score(input=input, target=target, num_classes=num_classes, average=average)