# site packages
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'   # shut down the warning msgs from tensorboard
import time
import torch
import argparse
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ignore the warning messages of torcheval caused by calculating f1-score
from contextlib import contextmanager
import logging

# custom files
import utils
from model import VGG
from dataset import Monkey_Faces
from metrics import compute_multiclass_f1_score

############################# Fix the random seeds
utils.setup_random_seed(42)
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
    labels[labels>=args.num_class] = 0

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

def train(args):
    """
    Model training entry
    """
    # define running device and pre-processing methods
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size), antialias=False),
        transforms.Normalize(mean=args.dataset_mean, std=args.dataset_std),
        transforms.RandomCrop(227, padding=0),
    ])
    
    # No validation dataset, split it from train/val dataset
    train_dataset = Monkey_Faces(args.train_path, class_id_mapping={}, transform=transform, dataset_type="train")
    class_id_mapping = train_dataset.class_id_mapping       # obtain the mapping relationship between class names and ids
    test_dataset = Monkey_Faces(args.test_path, class_id_mapping=class_id_mapping, transform=transform, dataset_type="test")
    # check the number of class
    if len(class_id_mapping)!=args.num_class:
        raise ValueError("The number of labels in training dataset and settings is different: dataset: {} | setting: {}".format(len(class_id_mapping), args.num_class))
    # obtain validation dataset
    if len(train_dataset)>len(test_dataset):        # enough training dataset, split training dataset. 
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset)-len(test_dataset), len(test_dataset)])
    else:                                           # Otherwise, split test dataset
        val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [len(test_dataset)-int(len(test_dataset)*0.5), int(len(test_dataset)*0.5)])

    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True), \
                                            DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True),   \
                                            DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    print("train set length: {} | val set length: {} | test set length: {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    # define model, loss function and optimizer
    model = VGG(nc=args.num_class, version=16, batch_size=args.batch_size, batch_normalize=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # create tensoboard log writer
    save_path = utils.query_dir_name(args.save_path)
    writer = SummaryWriter(save_path)

    print("training starts!")
    count = 0
    train_loader_length = len(train_loader)

    es_patience_count = 0
    lr_loss_history = []
    lr_patience_count = 0

    for epoch in range(0, args.epochs):
        model.train()
        train_loss = []         # global loss
        train_f1_score = []     # global f1 score

        epoch_correct = 0       # global correct number (accu.)
        start_point = time.perf_counter()   # time log
        for batch_index, (imgs, labels) in enumerate(train_loader):


            preds, n_correct, batch_loss, batch_f1_score = forward_one_batch(model, criterion, imgs, labels,
                                                                             device, args)

            # back propagation and update parameters
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # write batch results to tensorboard
            epoch_correct += n_correct
            train_loss.append(batch_loss.detach())
            train_f1_score.append(batch_f1_score.detach().cpu().numpy())
            writer.add_scalar("train_metrics/batch_train_loss", batch_loss.detach().cpu(), global_step=count)
            writer.add_scalar("train_metrics/batch_train_accuracy", n_correct/len(labels), global_step=count)
            writer.add_scalar("train_metrics/batch_train_f1", batch_f1_score, global_step=count)
            count += 1
            if batch_index%5==0:    # ensure the program is living or not
                print('\r'+"Epoch{} {}/{} | Accuracy: {:3f} | F1-score: {:3f}".format(epoch, count%train_loader_length, train_loader_length,
                                                                                      n_correct/len(labels), batch_f1_score), end='', flush=True)  
            
        #  write epoch results to tensorboard 
        epoch_accuracy = epoch_correct / (train_loader_length*args.batch_size)    # drop_last=True
        train_loss = torch.mean(torch.tensor(train_loss))
        writer.add_scalar("train_metrics/epoch_train_loss", train_loss, global_step=epoch)
        writer.add_scalar("train_metrics/epoch_train_accuracy", epoch_accuracy, global_step=epoch)
        writer.add_scalar("train_metrics/epoch_train_f1", np.average(train_f1_score), global_step=epoch)

        # print metrics after specific epochs
        if epoch%args.print_step==0:
            print("\rEpoch {}| Loss: {:.5f} | Accuracy: {:.3f} | F1-score: {:.3f} | time: {:.3f}s".format(epoch, train_loss, epoch_accuracy,
                                                                                            np.average(train_f1_score), time.perf_counter()-start_point), end='\n', flush=True)
        # save model, optimizer, metrics and other hyper-parameters after specific epochs
        if epoch%args.save_step==0:
            weight_save_dir = os.path.join(save_path, "weights")
            utils.save_model(weight_save_dir, model, optimizer, epoch, loss=train_loss, accu=epoch_accuracy,f1=np.average(train_f1_score), args=args)

        # validate the model
        if epoch%args.val_step==0:
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
                # writer.add_scalar("loss/epoch_validation", train_loss, global_step=epoch)
                writer.add_scalars("val_metrics/loss_comparison", {"train_loss": train_loss,
                                                    "validation_loss": val_loss}, global_step=epoch)
                writer.add_scalars("val_metrics/accuracy_comparison", {"train_accuracy": epoch_accuracy,
                                                                "val_accuracy": val_accuracy} , global_step=epoch)
                writer.add_scalars("val_metrics/f1_comparison", {"train_f1": np.average(train_f1_score),
                                                            "validation_f1": np.average(val_f1_score)}, global_step=epoch)
                print("\rValidation {} | Loss: {:.5f} | Accuracy: {:.3f} | F1-score: {:.3f}".format(epoch, val_loss, val_accuracy, np.average(val_f1_score)), end='\n', flush=True)
        ############################# Update learning rate here
        if len(lr_loss_history)<args.lr_step+1:
            lr_loss_history.append(train_loss)
        else:
            sum_loss_diff = sum([lr_loss_history[0]-lr_loss for lr_loss in lr_loss_history[1:]])    # the sum of loss in the list
            if sum_loss_diff > args.metric_threshold * args.lr_step:        # almost keep in same for epochs
                for group in optimizer.param_groups:
                    group["lr"] /= 10   # decrease learning rate
            lr_loss_history.pop(0)      # maintain list length
        #############################

        ############################# Determine early stop or not here
        if torch.abs(val_loss-train_loss) <= 1e-2:
            es_patience_count += 1
            if es_patience_count == args.es_patience_step:
                print("Performance without improvement for epochs, early stop in epoch %d"%epoch)
                break
        else:
            es_patience_count = 0
        #############################

    # Test the model
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
        # train and val: use the latest epoch results
        writer.add_scalars("test_metrics/loss_comparison", {"train_loss": train_loss, "validation_loss": val_loss,
                                                    "test_loss": test_loss}, global_step=epoch)
        writer.add_scalars("test_metrics/accuracy_comparison", {"train_accuracy": epoch_accuracy, "val_accuracy": val_accuracy,
                                                        "test_accuracy": test_accuracy} , global_step=epoch)
        writer.add_scalars("test_metrics/f1_comparison", {"train_f1": np.average(train_f1_score), "validation_f1": np.average(val_f1_score),
                                                    "test_f1": np.average(test_f1_score)}, global_step=epoch)
        print("Tests | Loss: {:.5f} | Accuracy: {:.3f} | F1-score: {:.3f}".format(test_loss, test_accuracy, np.average(test_f1_score)))

def argparser():
    parser = argparse.ArgumentParser(description="configurations for training")

    parser.add_argument("-trp", "--train_path", type=str, default=r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface", help="the path of training dataset")
    parser.add_argument("-tep", "--test_path", type=str, default=r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\test_Magface", help="the path of test dataset")
    parser.add_argument("-sp", "--save_path", type=str, default="./logs", help="the path to save files")

    parser.add_argument("--save_step", type=int, default=2, help="save the model to local after given epochs")
    parser.add_argument("--print_step", type=int, default=1, help="print the results after given epochs")
    parser.add_argument("--val_step", type=int, default=1, help="validate the model after given epochs. Set 0 to ignore validation.")
    parser.add_argument("--es_patience_step", type=int, default=5, help="stop training after given epochs without metric improvement")

    # Update learning rate
    parser.add_argument("--metric_threshold", type=float, default=5e-2, help="if loss reduction is less than this number, count plus 1")    
    parser.add_argument("--lr_step", type=int, default=3, help="decrease learning rate after given epochs without metric improvement")    

    parser.add_argument("-es", "--epochs", type=int, default=20, help="the number of training epochs")
    parser.add_argument("-nc", "--num_class", type=int, help="the number of category/class")
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="the number of batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-sz", "--img_size", type=int, default=256, help="the size of resized image")
    parser.add_argument("-d_mean", "--dataset_mean", type=tuple, default=(0.485, 0.383, 0.394), help="average values of monkey faces")
    parser.add_argument("-d_std", "--dataset_std", type=tuple, default=(0.051, 0.043, 0.052), help="standard deviation of monkey faces")
    
    parser.parse_args()
    return parser


if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()

    train(args)
    print("training finished")