import torch
from torchvision import transforms
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.img_size, antialias=False),
        transforms.RandomCrop(227, padding=0),
        # transforms.CenterCrop(args.img_size),
        # transforms.Normalize(mean=args.dataset_mean, std=args.dataset_std)
    ])
    
    # No validation dataset
    train_dataset = Monkey_Faces(args.train_path, transform=transform)
    test_dataset = Monkey_Faces(args.test_path, transform=transform)
    train_loader, test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True), \
                                DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = VGG(nc=args.num_class, version=16, batch_size=args.batch_size, batch_normalize=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    save_path = utils.mk_new_dir(args.save_path)
    writer = SummaryWriter(save_path)

    print("training begins!")
    count = 0
    train_loader_length = len(train_loader)
    for epoch in range(0, args.epochs):
        model.train()
        training_loss = []

        epoch_correct = 0
        start_point = time.perf_counter()
        for batch_index, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            current_iteration_correct = (torch.sum(preds == labels)).to('cpu').numpy()
            epoch_correct += current_iteration_correct

            optimizer.zero_grad()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss.append(loss.detach())
            writer.add_scalar("loss/batch_train", loss.detach().cpu(), global_step=count)
            writer.add_scalar("accu/iter_accuracy", current_iteration_correct/train_loader_length, global_step=count)
            count += 1
            print('\r'+"{}/{}".format(count%train_loader_length, train_loader_length), end='', flush=True)
            
        
        accuracy = epoch_correct / (train_loader_length*args.batch_size)    # drop_last=True
        training_loss = torch.mean(torch.tensor(training_loss))
        writer.add_scalar("loss/train", training_loss, global_step=epoch)
        writer.add_scalar("accu/epoch_accuracy", accuracy, global_step=epoch)

        if epoch%args.print_step==0:
            print("Epoch {}| Loss: {:.5f} | Accuracy: {:.3f} | time: {:.3f}s".format(epoch, training_loss, accuracy,
                                                                                     time.perf_counter()-start_point))
        if epoch%args.save_step==0 and epoch!=0:
            weight_save_dir = os.path.join(save_path, "weights")
            utils.save_model(weight_save_dir, model, optimizer, epoch, loss=training_loss, accu=accuracy, args=args)
        
        weight_save_dir = os.path.join(save_path, "final_weights")
        utils.save_model(weight_save_dir, model, optimizer, epoch, loss=training_loss, accu=accuracy, args=args)


def argparser():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description="configurations for training")

    # parser.add_argument("-trp", "--train_path", type=str, default=r"E:\datasets\monkeys\demo")
    # parser.add_argument("-tep", "--test_path", type=str, default=r"E:\datasets\monkeys\demo")
    parser.add_argument("-trp", "--train_path", type=str, default=r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface")
    parser.add_argument("-tep", "--test_path", type=str, default=r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface")
    parser.add_argument("-sp", "--save_path", type=str, default="./logs")

    parser.add_argument("--save_step", type=int, default=100)
    parser.add_argument("--print_step", type=int, default=1)


    parser.add_argument("-es", "--epochs", type=int, default=300)
    parser.add_argument("-nc", "--num_class", type=int, default=122)
    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-sz", "--img_size", type=int, default=256)
    parser.add_argument("-d_mean", "--dataset_mean", type=tuple, default=(0.4896, 0.4305, 0.4188), help="Values of Monkey faces")
    # parser.add_argument("-d_std", "--dataset_std", default=(0.299, 0.224, 0.255))
    parser.add_argument("-d_std", "--dataset_std", type=tuple, default=(0.299, 0.224, 0.255), help="Values of ImageNet, replacement needed")
    

    parser.parse_args()
    return parser

if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()

    train(args)
    print("training stops")