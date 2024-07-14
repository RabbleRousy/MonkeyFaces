import os
import cv2
import math
import shutil
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False

def mk_new_dir(root_dir):
    dir_count = len(os.listdir(root_dir))
    dir_name = os.path.join(root_dir, "exp"+str(dir_count))
    if not create_dir(dir_name):
        dir_name = os.path.join(root_dir, "exp"+str(len(os.listdir(root_dir))+1))
    return dir_name

def save_model(path, model, optimizer, epoch, loss, psnr, args):
    create_dir(path)
    results = {
        "epoch": epoch,
        "best_loss": loss,
        "best_psnr": psnr,
        "fine_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": args._get_kwargs()
    }
    torch.save(results, os.path.join(path, "{}_{}.pth.tar".format(epoch, psnr)))

def save_model(path, model, optimizer, epoch, loss, accu, args):
    create_dir(path)
    results = {
        "epoch": epoch,
        "best_loss": loss,
        "accuracy": accu,
        "fine_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": args._get_kwargs()
    }
    torch.save(results, os.path.join(path, "{}_{}.pth.tar".format(epoch, accu)))

def copy2dir():
    # load & save paths
    dataset_path = r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface"
    save_path_root = r"E:\temp"
    # dirs of all monkeys
    dirs = os.listdir(dataset_path)

    # the number of images under each individual
    threshold = 200
    for index, dir in enumerate(dirs):
        # dir of each monkey
        dir_path = os.path.join(dataset_path, dir)
        # the number of images in each individual file
        n_imgs = len(os.listdir(dir_path))
        # meet requirement, copy to destination
        if n_imgs<threshold:
            save_path = os.path.join(save_path_root, dir)
            # if destination dir does not exist, mkdir
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # copy2dir
            for img in os.listdir(dir_path):
                shutil.copy2(os.path.join(dir_path, img), os.path.join(save_path, img))
        print("\r"+"{}/{}".format(index, len(dirs)), end='', flush=True)
    print("Finished")

def cal_dataset(root, new_size=(227, 227)):
    img_paths = []
    monkey_dirs = os.listdir(root)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(new_size)])  # transforms.CenterCrop((227, 227)),
    for dir in monkey_dirs:
        dir_path = os.path.join(root, dir)
        for img_name in os.listdir(dir_path)[:5]:
            img_path = os.path.join(dir_path, img_name)
            if ".simDB" not in img_path:
                img_paths.append(img_path)
    
    img_total = None
    for img_path in tqdm(img_paths):
        if not img_path:
            continue
        try:
            img = transform(cv2.imread(img_path))
        except Exception as e:
            print(e)
            print(img_path)

        if img_total is None:
            img_total = torch.zeros_like(img)

        img_total = img_total + img
        
    img_total = img_total/len(img_paths)
    avg_RGB = torch.mean(img_total, dim=0)
    std_RGB = torch.std(img_total, dim=0)
    cv2.imwrite("total_img.jpg", img=img_total.numpy().transpose([1,2,0])*255)
    cv2.imwrite("avg_img.jpg", img=avg_RGB.numpy().transpose([1,2,0])*255)
    cv2.imwrite("std_img.jpg", img=std_RGB.numpy().transpose([1,2,0])*255)
    # print("Total pixel values in BGR: {}".format(img_total))
    # print("Avg. R: {}, G: {}, B:{}".format(avg_R, avg_G, avg_B))
    # print("Std. R: {}, G: {}, B:{}".format(std_R, std_G, std_B))
    # Training set:
    # Avg. R: 0.4895858355920331, G: 0.43052453455622913, B:0.41882570491399984
    # BUG Std. R: 0.006878355312770115, G: 0.018366679084975822, B:0.010428968049148573


if __name__ == "__main__":
    # copy2dir()
    new_size = (227, 227)
    root = r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface"
    cal_dataset(root, new_size)

