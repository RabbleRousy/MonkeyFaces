import torch

############################# This file includes several training tricks
# 1. Learning rate scheduler
# 2. Early stopping 
#############################

class LRScheduler(object):
    def __init__(self, optimizer, initial_lr=1e-3, patience=3, factor=0.1, min_lr=1e-6):
        self.optimizer = optimizer
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_loss = float("inf")
        self.waiting_epoch = 0

    def update_lr(self):
        if self.current_lr < self.min_lr:
            print("Minimal learning rate, no update")
            return
        for params in self.optimizer.param_groups:
            params["lr"] = self.current_lr * self.factor
        print("Learning rate is updated to {}".format(self.current_lr))

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.waiting_epoch = 0
        else:
            self.waiting_epoch += 1

        if self.waiting_epoch >= self.patience:
            self.update_lr()
            self.waiting_epoch = 0
    

class EarlyStopping(object):
    def __init__(self, patience=5, delta=5e-2) -> None:
        self.patience = patience
        self.delta = delta
        self.waiting_epoch = 0
        self.best_loss_diff = float("inf")

    def __call__(self, train_loss, val_loss):
        current_loss_diff = torch.abs(train_loss-val_loss) + self.delta
        if current_loss_diff < self.best_loss_diff:
            self.best_loss_diff = current_loss_diff
            self.waiting_epoch = 0
        else:
            self.waiting_epoch += 1
        if self.waiting_epoch >= self.patience:
            print("No improvements for %d epochs, early stopping"%self.patience)
            exit("Early stopping")
        