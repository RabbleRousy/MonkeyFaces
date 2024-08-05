import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader

from model import VGG
from utils import load_model
from dataset import Monkey_Faces
from metrics import MetricLogger
from train import test

if __name__ == "__main__":

    # configurations
    nc = 122
    # nc = 122 #              # this parameter determines the output layer
    l_r = 1e-3
    img_size = 227          # we used transforms.RandomCrop(227) while training
    batch_size = 128
    batch_normalize = True  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=False)
    ])

    test_img_dir = r"E:\ws\MonkeyFace\Prototypes\Zhang\runs\demotest"
    # test_img_dir = r"E:\ws\MonkeyFace\Prototypes\Zhang\runs\demoset"
    # test_img_dir = r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\test_Magface"
    train_dataset_path = r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface"
    # train_dataset_path = r"E:\datasets\monkeys\demo"
    model_path = r"E:\ws\MonkeyFace\Prototypes\Zhang\logs\exp5\weights\12_0.9711587778429074.pth.tar"
    # obtain id and class mapping
    train_dataset = Monkey_Faces(train_dataset_path, class_id_mapping={})
    class_id_mapping = train_dataset.class_id_mapping

    test_dataset = None
    test_true_class_id = None
    if "demo" not in test_img_dir:
        test_dataset = Monkey_Faces(test_img_dir, class_id_mapping=class_id_mapping, transform=transform, dataset_type="test")
        test_true_class_id = test_dataset.true_class_id

    if not test_dataset:
        # load data
        img_names = [name for name in sorted(os.listdir(test_img_dir))] # also labels
        imgs = [transform(cv2.imread(os.path.join(test_img_dir, name))) for name in img_names]  # preprocessing images
        img_tensor = torch.stack(imgs, dim=0)       # transform images to tensors

        # define model(required) and optimizer(optional)
        model = VGG(nc=nc, version=16, batch_size=len(imgs), batch_normalize=True, initialize=False)
        # optimizer = torch.optim.Adam(model.parameters(), l_r)
        model, optimizer = load_model(model_path, model, )
        # print(summary(model))
        device = "cuda" if torch.cuda.is_available() else "cpu"     # device (cpu/gpu)
        results = []                        # list storing prediction results
        model = model.to(device)            # load model to device
        inputs = img_tensor.to(device)

        with torch.no_grad():
            model.eval()
            outputs = model(inputs)             # run the vgg
            _, preds = torch.max(outputs, 1)    # transform the outputs to class id

        class_id_tuple = tuple(class_id_mapping.items())
        results.extend([class_id_tuple[class_id][0] for class_id in preds.detach().cpu().numpy()])   # load class-ids to cpu 

        # visualize the results through matplotlib
        n_cols = 5                                      # the number of columns
        n_rows = int(np.ceil(len(img_names)//n_cols))   # the number of rows
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 15))   # initialize canvas
        count = 0   # select images from list
        for row in range(n_rows):
            for col in range(n_cols):
                # img = imgs[count].numpy().transpose([1,2,0])
                axs[row][col].imshow(imgs[count].numpy().transpose([1,2,0])[..., ::-1])    # transform image shape from (3,227,227') to (227,227',3)
                axs[row][col].axis('off')                                       # shut down axis

                # simplify sub-figure titles
                ground_truth_name = "'".join(img_names[count].split('x')[3:-1])
                if not ground_truth_name:
                    ground_truth_name = img_names[count].split('.')[0]
                pred_name = results[count]

                # axs[row][col].set_title("T:{}\nP:{}".format(img_names[count][:-4], results[count]))
                axs[row][col].set_title("T:{}\nP:{}".format(ground_truth_name, results[count]))
                count += 1

        plt.tight_layout()
        plt.show()

    else: 
        test_loader = DataLoader(test_dataset, batch_sampler=batch_size, shuffle=True, drop_last=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = VGG(nc=nc, version=16, batch_size=batch_size, batch_normalize=True, initialize=False)
        model, optimizer = load_model(model_path, model, )
        model = model.to(device)

        test_category_metric_logger = MetricLogger(len(test_true_class_id))
        
        with torch.no_grad():
            model.eval()
            