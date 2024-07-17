import torch
import torch.utils
from torchvision import transforms
from torcheval.metrics.functional import multiclass_f1_score
import numpy as np
import argparse
from model import VGG
from dataset import Monkey_Faces
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import utils
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'
import time

def train(args):
    # define running device and pre-processing methods
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.img_size, args.img_size), antialias=False),
        transforms.Normalize(mean=args.dataset_mean, std=args.dataset_std),
        transforms.RandomCrop(227, padding=0),
    ])
    
    # No validation dataset, split it from train/val dataset
    train_dataset = Monkey_Faces(args.train_path, id_class_mapping={}, transform=transform)
    id_class_mapping = train_dataset.id_class_mapping       # obtain the mapping relationship between class names and ids
    test_dataset = Monkey_Faces(args.test_path, id_class_mapping=id_class_mapping, transform=transform)

    # obtain validation dataset
    if len(train_dataset)>len(test_dataset):        # enough training dataset, split training dataset. Otherwise, split test dataset
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset)-len(test_dataset), len(test_dataset)])
    else:
        val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [len(test_dataset)-int(len(test_dataset)*0.5), int(len(test_dataset)*0.5)])

    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True), \
                                            DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True),   \
                                            DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # define model, loss function and optimizer
    model = VGG(nc=args.num_class, version=16, batch_size=args.batch_size, batch_normalize=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # create tensoboard log writer
    save_path = utils.query_dir_name(args.save_path)
    writer = SummaryWriter(save_path)

    print("training starts!")
    count = 0
    train_loader_length = len(train_loader)

    for epoch in range(0, args.epochs):
        model.train()
        # store metric values for each epoch
        training_loss = []
        training_f1 = []

        epoch_correct = 0
        start_point = time.perf_counter()   # time log
        for batch_index, (imgs, labels) in enumerate(train_loader):
            # load data to device (cpu/gpu)
            imgs = imgs.to(device)
            labels = labels.to(device)

            # forward propagation and transform probability to class id 
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            # calculate accuracy
            current_iteration_correct = (torch.sum(preds == labels)).to('cpu').numpy()
            epoch_correct += current_iteration_correct
            # calculate f1 score
            train_f1_score = multiclass_f1_score(input=preds, target=labels, 
                                                 num_classes=args.num_class, average="macro")

            # back propagation and update parameters
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # write information to tensorboard logs
            training_loss.append(loss.detach())
            training_f1.append(train_f1_score.detach())
            writer.add_scalar("loss/batch_train", loss.detach().cpu(), global_step=count)
            writer.add_scalar("metrics/batch_accuracy", current_iteration_correct/train_loader_length, global_step=count)
            writer.add_scalar("metrics/batch_f1", train_f1_score, global_step=count)
            count += 1
            print('\r'+"{}/{}".format(count%train_loader_length, train_loader_length), end='', flush=True)  # to ensure the program is living or not
            
        # calculate epoch metrics
        accuracy = epoch_correct / (train_loader_length*args.batch_size)    # drop_last=True
        training_loss = torch.mean(torch.tensor(training_loss))
        writer.add_scalar("loss/epoch_train", training_loss, global_step=epoch)
        writer.add_scalar("metrics/epoch_accuracy", accuracy, global_step=epoch)
        writer.add_scalar("metrics/epoch_f1", torch.mean(training_f1), global_step=epoch)

        # print metrics after specific steps
        if epoch%args.print_step==0:
            print("\rEpoch {}| Loss: {:.5f} | Accuracy: {:.3f} | F1-score: {:.3f} | time: {:.3f}s".format(epoch, training_loss, accuracy,
                                                                                            train_f1_score, time.perf_counter()-start_point))
        # save model, optimizer, metrics and other hyper-parameters after specific steps
        if epoch%args.save_step==0 and epoch!=0:
            weight_save_dir = os.path.join(save_path, "weights")
            utils.save_model(weight_save_dir, model, optimizer, epoch, loss=training_loss, accu=accuracy,f1=train_f1_score, args=args)

def argparser():
    parser = argparse.ArgumentParser(description="configurations for training")

    parser.add_argument("-trp", "--train_path", type=str, default=r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface", help="the path of training dataset")
    parser.add_argument("-tep", "--test_path", type=str, default=r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\test_Magface", help="the path of test dataset")
    parser.add_argument("-sp", "--save_path", type=str, default="./logs", help="the path to save files")

    parser.add_argument("--save_step", type=int, default=5, help="save the model to local after given epochs")
    parser.add_argument("--print_step", type=int, default=1, help="print the results after given steps")
    parser.add_argument("--val_test", action="store_true", help="whether validate or test the model or not")

    parser.add_argument("-es", "--epochs", type=int, default=300, help="the number of training epochs")
    parser.add_argument("-nc", "--num_class", type=int, default=122, help="the number of category/class")
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help="the number of batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-sz", "--img_size", type=int, default=256, help="the size of resized image")
    parser.add_argument("-d_mean", "--dataset_mean", type=tuple, default=(0.485, 0.383, 0.394), help="average values of monkey faces")
    parser.add_argument("-d_std", "--dataset_std", type=tuple, default=(0.051, 0.043, 0.052), help="Values of ImageNet, replacement needed")
    

    parser.parse_args()
    return parser

if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()

    train(args)
    print("training stops")