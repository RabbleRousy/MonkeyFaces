# site packages
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'   # shut down the warning msgs from tensorboard
import time
import torch
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# ignore the warning messages of torcheval caused by calculating f1-score
from contextlib import contextmanager
import logging

# custom files
from model import VGG
from dataset import load_dataset
from tricks import EarlyStopping, LRScheduler
from metrics import compute_multiclass_f1_score
from utils import save_model, load_model, query_dir_name, setup_random_seed

############################# Fix the random seeds
setup_random_seed(42)
#############################

@contextmanager
def suppress_warnings():
    """
    Shut down warning messages.
    """
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)

def forward_one_batch(model, criterion, imgs, labels, device, args):
    """
    Forward a batch of data.
    Return:
    >>> preds: prediction results
    >>> n_correct: correct samples in this batch
    >>> loss: loss of this batch
    >>> f1_score: f1_score of this batch
    """
    # load data to device (cpu/gpu)
    imgs = imgs.to(device)
    labels = labels.to(device)

    # forward propagation and transform probability to class id 
    outputs = model(imgs)
    _, preds = torch.max(outputs, 1)

    # avoid label errors
    # BUG, leading a small f1 score in validation and test set if using partial training set
    # labels[labels>=args.num_class] = 0

    # loss
    loss = criterion(outputs, labels)
    # the number of correct samples
    n_correct = (torch.sum(preds == labels)).to('cpu').numpy()
    # f1 score. Ignore the warning messages which pollute the echo
    with suppress_warnings():
        f1_score = compute_multiclass_f1_score(input=preds, target=labels, 
                                        num_classes=args.num_class, average="macro")
    
    ############################# For small gpu memory device
    # If you have enough gpu memory, please comment this code
    # del imgs
    # del labels
    # torch.cuda.empty_cache()
    #############################

    return preds, n_correct, loss, f1_score


def train(args, model, optimizer, criterion, device, writer, train_loader, val_loader):
    """
    Model training entry
    """

    print("training starts!")
    count = 0
    train_loader_length = len(train_loader)

    scheduler = LRScheduler(optimizer, initial_lr=args.learning_rate, patience=args.lr_step)
    early_stopping = EarlyStopping(args.es_patience_step)

    for epoch in range(0, args.epochs):
        model.train()
        train_loss = []         # global loss
        train_f1_score = []     # global f1 score

        epoch_correct = 0       # global correct number (accu.)
        start_point = time.perf_counter()   # time log
        for batch_index, (imgs, labels) in enumerate(train_loader):

            preds, n_correct, batch_loss, batch_f1_score = forward_one_batch(model, criterion, imgs, labels, device, args)

            # back propagation and update parameters
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # write batch results to tensorboard
            epoch_correct += n_correct
            train_loss.append(batch_loss.detach())
            train_f1_score.append(batch_f1_score.detach().cpu().numpy())
            writer.add_scalar("train_metrics/batch_train_loss",     batch_loss.detach().cpu(), global_step=count)
            writer.add_scalar("train_metrics/batch_train_accuracy", n_correct/len(labels),     global_step=count)
            writer.add_scalar("train_metrics/batch_train_f1",       batch_f1_score,            global_step=count)
            count += 1
            if batch_index%5==0:    # ensure the program is living or not
                print('\r'+"Epoch{} {}/{} | Accuracy: {:3f} | F1-score: {:3f}".format(epoch, count%train_loader_length, train_loader_length,
                                                                                      n_correct/len(labels), batch_f1_score), end='', flush=True)  
        #  write epoch results to tensorboard 
        train_f1_score = np.average(train_f1_score)
        epoch_accuracy = epoch_correct / (train_loader_length*args.batch_size)    # drop_last=True
        train_loss = torch.mean(torch.tensor(train_loss))
        writer.add_scalar("train_metrics/epoch_train_loss",     train_loss,     global_step=epoch)
        writer.add_scalar("train_metrics/epoch_train_accuracy", epoch_accuracy, global_step=epoch)
        writer.add_scalar("train_metrics/epoch_train_f1",       train_f1_score, global_step=epoch)

        # print metrics after specific epochs
        if epoch%args.print_step==0:
            print("\rEpoch {}| Loss: {:.5f} | Accuracy: {:.3f} | F1-score: {:.3f} | time: {:.3f}s".format(epoch, train_loss, epoch_accuracy,
                                                                                            train_f1_score, time.perf_counter()-start_point), end='\n', flush=True)
        # save model, optimizer, metrics and other hyper-parameters after specific epochs
        if epoch%args.save_step==0:
            weight_save_dir = os.path.join(args.save_path, "weights")
            save_model(weight_save_dir, model, optimizer, epoch, loss=train_loss, accu=epoch_accuracy,f1=train_f1_score, args=args)

        # validate the model
        val_loss, val_accuracy, val_f1_score = None, None, None
        if epoch%args.val_step==0 and args.val:
            val_loss, val_accuracy, val_f1_score = validation(args, model, criterion, device, writer, val_loader, epoch, metrics=[train_loss, epoch_accuracy, train_f1_score])
        ############################# Update learning rate here
        scheduler.step(train_loss)
        #############################

        ############################# early stop
        if not val_loss:
            early_stopping(train_loss, val_loss=0)
        else:
            early_stopping(train_loss, val_loss)
        #############################
    return model, writer, [[train_loss, val_loss], [epoch_accuracy, val_accuracy], [train_f1_score, val_f1_score]]

def validation(args, model, criterion, device, writer, val_loader, epoch, metrics):
    with torch.no_grad():
        model.eval()
        val_loss = []
        val_f1_score = []
        val_correct = 0
        for batch_index, (imgs, labels) in enumerate(val_loader):
            preds, n_correct, batch_loss, batch_f1_score = forward_one_batch(model, criterion, imgs, labels,
                                                                        device, args)
            val_correct += n_correct
            val_loss.append(batch_loss.detach())
            val_f1_score.append(batch_f1_score.detach().cpu().numpy())
            print("\rValidating... {}/{}".format(batch_index, len(val_loader)), end='', flush=True)
        val_loss = torch.mean(torch.tensor(val_loss))
        val_accuracy = val_correct/(len(val_loader)*args.batch_size)
        val_f1_score = np.average(val_f1_score)
        writer.add_scalars("val_metrics/loss_comparison",     {"train_loss": metrics[0], "validation_loss": val_loss},       global_step=epoch)
        writer.add_scalars("val_metrics/accuracy_comparison", {"train_accuracy": metrics[1], "val_accuracy": val_accuracy} , global_step=epoch)
        writer.add_scalars("val_metrics/f1_comparison",       {"train_f1": metrics[2], "validation_f1": val_f1_score},       global_step=epoch)
        print("\rValidation {} | Loss: {:.5f} | Accuracy: {:.3f} | F1-score: {:.3f}".format(epoch, val_loss, val_accuracy, val_f1_score), end='\n', flush=True)
    return val_loss, val_accuracy, val_f1_score

def test(args, model, criterion, device, writer, test_loader, metrics):
    with torch.no_grad():
        print("Testing: ")
        model.eval()
        test_loss = []
        test_f1_score = []
        test_correct = 0
        for batch_index, (imgs, labels) in enumerate(test_loader):
            preds, n_correct, batch_loss, batch_f1_score = forward_one_batch(model, criterion, imgs, labels,
                                                                            device, args)
            test_correct += n_correct
            test_loss.append(batch_loss.detach())
            test_f1_score.append(batch_f1_score.detach().cpu().numpy())
        test_loss = torch.mean(torch.tensor(test_loss))
        test_accuracy = test_correct/(len(test_loader)*(batch_index+1))
        test_f1_score = np.average(test_f1_score)
        # train and val: use the latest epoch results
        writer.add_scalars("test_metrics/loss_comparison",     {"train_loss": metrics[0][0],     "validation_loss": metrics[0][1], "test_loss": test_loss})
        writer.add_scalars("test_metrics/accuracy_comparison", {"train_accuracy": metrics[1][0], "val_accuracy": metrics[1][1],    "test_accuracy": test_accuracy})
        writer.add_scalars("test_metrics/f1_comparison",       {"train_f1": metrics[2][0],       "validation_f1": metrics[2][1],   "test_f1": test_f1_score})
        print("Test | Loss: {:.5f} | Accuracy: {:.3f} | F1-score: {:.3f}".format(test_loss, test_accuracy, test_f1_score))

def argparser():
    parser = argparse.ArgumentParser(description="configurations for training")

    parser.add_argument("-trp", "--train_path", type=str, default=r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface", help="the path of training dataset")
    parser.add_argument("-tep", "--test_path", type=str, default=r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\test_Magface", help="the path of test dataset")
    parser.add_argument("-sp", "--save_path", type=str, default="./logs", help="the path to save files")
    parser.add_argument("--checkpoint", type=str, default=None, help="load local model parameters for tuning")

    parser.add_argument("--save_step", type=int, default=2, help="save the model to local after given epochs")
    parser.add_argument("--print_step", type=int, default=1, help="print the results after given epochs")
    parser.add_argument("--val_step", type=int, default=1, help="validate the model after given epochs. Set 0 to ignore validation.")
    parser.add_argument("--es_patience_step", type=int, default=5, help="stop training after given epochs without metric improvement")
    parser.add_argument("--lr_step", type=int, default=3, help="decrease learning rate after given epochs without metric improvement")    

    parser.add_argument("-es", "--epochs", type=int, default=20, help="the number of training epochs")
    parser.add_argument("-nc", "--num_class", type=int, help="the number of category/class")    # 122 / 28
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="the number of batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-sz", "--img_size", type=int, default=256, help="the size of resized image")
    parser.add_argument("-d_mean", "--dataset_mean", type=tuple, default=(0.485, 0.383, 0.394), help="average values of monkey faces")
    parser.add_argument("-d_std", "--dataset_std", type=tuple, default=(0.051, 0.043, 0.052), help="standard deviation of monkey faces")

    parser.add_argument("--train", action="store_false",  help="train or not")
    parser.add_argument("--val", action="store_true",  help="using validation set or not")
    parser.add_argument("--test", action="store_true", help="using test set or not")

    parser.parse_args()
    return parser


if __name__ == "__main__":
    # load settings
    parser = argparser()
    args = parser.parse_args()
    
    # load dataset
    train_loader, val_loader, test_loader = load_dataset(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # execution device
    model = VGG(nc=args.num_class, version=16, batch_size=args.batch_size, batch_normalize=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    # load from checkpoint
    if args.checkpoint:
        model, optimizer = load_model(args.checkpoint, model, optimizer)

    # create tensoboard log writer
    args.save_path = query_dir_name(args.save_path)
    writer = SummaryWriter(args.save_path)

    # train and test
    if args.train:
        model, writer, metrics = train(args, model, optimizer, criterion, device, writer, train_loader, val_loader)
        print("Training finished")
    if args.test:
        if args.test and not args.train and not args.checkpoint:
            print("WARNING: No pre-trained model. Test on randomly initialized model.")
        test(args, model, criterion, device, writer, test_loader, metrics)
    print("Finished")