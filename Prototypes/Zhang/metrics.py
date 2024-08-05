import torch
from torcheval.metrics.functional import multiclass_f1_score, multiclass_precision, multiclass_recall

# class MetricLogger(object):
#     def __init__(self, n_category) -> None:
#         self.n_category = n_category
#         self.category_statics = torch.tensor([])    # (n_epoch, nc, 3)
#         self.category_metrics = torch.tensor([])    # (n_epoch, nc, 3)
#         self.temp_category_count = torch.zeros(size=(self.n_category, 3), dtype=torch.int16)

#     def count_confusion_matrix(self, preds, labels):
#         """
#         Count the number of TP, FP, FN for each category.
#         You should call this function after each epoch.
#         """
#         # self.category_statics.append([[0,0,0] for _ in range(self.n_category)])
#         mask = (preds==labels)*preds
#         # print(mask)
#         # exit()
#         for category in range(self.n_category):
#             n_TP = torch.count_nonzero(mask==category)
#             n_FP = max(0, torch.count_nonzero(labels==category) -n_TP)
#             n_FN = max(0, torch.count_nonzero(preds==category)  -n_TP)
#             self.temp_category_count[category][0] += n_TP
#             self.temp_category_count[category][1] += n_FP
#             self.temp_category_count[category][2] += n_FN

#     def concat_count_matrix(self):
#         self.category_statics = torch.vstack([self.category_statics, self.temp_category_count]) if len(self.category_statics)>1 else self.temp_category_count
#         self.temp_category_count = torch.zeros(size=(self.n_category, 3))

#     def compute_metrics(self):
#         """
#         Compute precision, recall and f1-score for each category under each epoch.
#         """
#         precision = self.category_statics[..., 0] / (self.category_statics[..., 0] + self.category_statics[..., 1])
#         recall    = self.category_statics[..., 0] / (self.category_statics[..., 0] + self.category_statics[..., 2])
#         f1_score  = 2*precision*recall / (precision+recall)
#         torch.nan_to_num_(precision, 0)
#         torch.nan_to_num_(recall, 0)
#         torch.nan_to_num_(f1_score, 1)
#         self.category_metrics = torch.concat([precision, recall, f1_score], dim=-1).reshape((-1, 3))


class MetricLogger(object):
    def __init__(self, n_category) -> None:
        self.n_category = n_category
        self.preds = []
        self.labels = []
        self.category_metrics = []

    def count_confusion_matrix(self, preds, labels):
        """
        Count the number of TP, FP, FN for each category.
        You should call this function after each epoch.
        """
        self.preds = torch.concat([self.preds, preds], dim=-1) if len(self.preds)>1 else preds
        self.labels = torch.concat([self.labels, labels], dim=-1) if len(self.labels)>1 else labels

    def concat_count_matrix(self):
        # self.category_statics = torch.vstack([self.category_statics, self.temp_category_count]) if len(self.category_statics)>1 else self.temp_category_count
        # self.temp_category_count = torch.zeros(size=(self.n_category, 3))
        pass

    def compute_metrics(self):
        """
        Compute precision, recall and f1-score for each category under each epoch.
        """
        precision = multiclass_precision(self.preds, self.labels, num_classes=self.n_category, average=None)
        recall = multiclass_recall(self.preds, self.labels, num_classes=self.n_category, average=None)
        f1_score = compute_multiclass_f1_score(self.preds, self.labels, num_classes=self.n_category, average=None)
        self.category_metrics = torch.concat([f1_score, precision, recall], dim=-1).reshape((-1, 3))


# def compute_confusion_matrix(class_num_count, preds, labels, n_category:int):
#     """
#     Calculate the metrics for each category.
#     """
#     # count the number of TP, FP and FN for each category
#     for category in range(n_category):
#         n_TP = torch.count_nonzero(preds==labels)
#         n_FP = max(0, torch.count_nonzero(labels==category)-n_TP)
#         n_FN = max(0, torch.count_nonzero(preds==category) -n_TP)
#         class_num_count[category][0] += n_TP
#         class_num_count[category][1] += n_FP
#         class_num_count[category][2] += n_FN

#     return class_num_count

# def compute_precision_recall_f1(TP, FP, FN):
#     """
#     Calculate f1-score for each class.
#     """
#     precision = TP / (TP+FP)
#     recall = TP / (TP+FN)
#     F1 = 2*precision*recall / (precision+recall)
#     return [precision, recall, F1]

def compute_multiclass_f1_score(input, target, num_classes, average):
    """
    Use the function from site-package to calculate f1-score for multi-class.
    """
    return multiclass_f1_score(input=input, target=target, num_classes=num_classes, average=average)