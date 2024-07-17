import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from model import VGG
from dataset import Monkey_Faces
############################# Load model (pre-trained model or models trained from scratch)
def load_model(model_path, model, optimizer=None):
    model_parameters = torch.load(model_path)
    model.load_state_dict(model_parameters['fine_model'])
    if optimizer:
        optimizer.load_state_dict(model_parameters['optimizer'])
    return model, optimizer
############################# Load model


############################# Predict
if __name__ == "__main__":

    # configurations
    nc = 122 # /29          # this parameter determines the output layer
    l_r = 1e-3
    img_size = 227          # we used transforms.RandomCrop(227) while training
    batch_size = 128
    batch_normalize = True  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=False)
    ])

    test_img_dir = r"E:\ws\MonkeyFace\Prototypes\Zhang\runs"
    train_dataset_path = r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface"
    model_path = r"E:\ws\MonkeyFace\Prototypes\Zhang\logs\exp1\final_weights\10_0.9462433880846325.pth.tar"
    
    # obtain id and class mapping
    train_dataset = Monkey_Faces(train_dataset_path, id_class_mapping={})
    id_class_mapping = train_dataset.id_class_mapping

    # load data
    img_names = [name for name in sorted(os.listdir(test_img_dir))] # also labels
    imgs = [transform(cv2.imread(os.path.join(test_img_dir, name))) for name in img_names]  # preprocessing images
    img_tensor = torch.stack(imgs, dim=0)       # transform images to tensors

    # define model(required) and optimizer(optional)
    model = VGG(nc=122, version=16, batch_size=len(imgs), batch_normalize=True, initialize=False)
    # optimizer = torch.optim.Adam(model.parameters(), l_r)
    model, optimizer = load_model(model_path, model, )
    device = "cuda" if torch.cuda.is_available() else "cpu"     # device (cpu/gpu)

    results = []                        # list storing prediction results
    model = model.to(device)            # load model to device
    inputs = img_tensor.to(device)

    outputs = model(inputs)             # run the vgg
    _, preds = torch.max(outputs, 1)    # transform the outputs to class id

    results.extend([id_class_mapping[class_id] for class_id in preds.detach().cpu().numpy()])   # load class-ids to cpu 

    # visualize the results through matplotlib
    n_cols = 5                                      # the number of columns
    n_rows = int(np.ceil(len(img_names)//n_cols))   # the number of rows
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(30, 30))   # initialize canvas
    count = 0   # select images from list
    for row in range(n_rows):
        for col in range(n_cols):
            axs[row][col].imshow(imgs[count].numpy().transpose([1,2,0]))    # transform image shape from (3,227,227') to (227,227',3)
            axs[row][col].axis('off')                                       # shut down axis

            # simplify sub-figure titles
            # ground_truth_name = "'".join(img_names[count].split('x')[3:-1])
            # if not ground_truth_name:
            #     ground_truth_name = img_names[count].split('.')[0]
            # pred_name = results[count]
            axs[row][col].set_title("G:{}\nP:{}".format(img_names[count][:-4], results[count]))
            count += 1

    plt.tight_layout()
    plt.show()
