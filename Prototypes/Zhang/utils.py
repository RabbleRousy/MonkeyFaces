import os
import torch
import random
import shutil
import numpy as np

def setup_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dir(path):
    """
    Create directory if it does not exist
    """
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False

def query_dir_name(root_dir):
    """
    Obtain the name of new directory
    """
    dir_count = len(os.listdir(root_dir))
    dir_name = os.path.join(root_dir, "exp"+str(dir_count))
    if not create_dir(dir_name):
        dir_name = os.path.join(root_dir, "exp"+str(len(os.listdir(root_dir))+1))
    return dir_name

def save_model(path, model, optimizer, epoch, loss, accu, f1, args):
    """
    Save model and other training parameter to local
    """
    try:
        create_dir(path)
        results = {
            "epoch": epoch,
            "best_loss": loss,
            "accuracy": accu,
            "f1 score": f1,
            "final_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": args._get_kwargs()
        }
        torch.save(results, os.path.join(path, "{}_{}.pth.tar".format(epoch, accu)))
        return True
    except Exception as e:
        print(e)
        return False

def load_model(model_path, model, optimizer=None):
    """
    Load local model parameters
    """
    model_parameters = torch.load(model_path)
    model.load_state_dict(model_parameters['final_model'])
    if optimizer:
        optimizer.load_state_dict(model_parameters['optimizer'])
    return model, optimizer

def save_tensor(path, tensor):
    results = dict()
    if len(tensor.shape) == 2:
        tensor = tensor[None, ...]
    results["matrix"] = tensor
    torch.save(results, path)

def load_tensor(path):
    return torch.load(path)

def save2txt(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as fp:
        if not isinstance(content, str):
            content = str(content)
        fp.write(content)

def copy2dir():
    """
    Copy directories(categories) that meet the requirements to specific path
    """
    # load & save paths
    dataset_path = r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface"
    save_path_root = r"E:\datasets\monkeys\partial_train_v2"
    # dirs of all monkeys
    dirs = os.listdir(dataset_path)

    # the number of images under each individual
    threshold = 2800
    for index, dir in enumerate(dirs):
        # dir of each monkey
        dir_path = os.path.join(dataset_path, dir)
        # the number of images in each individual file
        n_imgs = len(os.listdir(dir_path))
        # meet requirement, copy to destination
        if n_imgs>threshold and n_imgs!=0:
            save_path = os.path.join(save_path_root, dir)
            # if destination dir does not exist, mkdir
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # copy2dir
            for img in os.listdir(dir_path):
                shutil.copy2(os.path.join(dir_path, img), os.path.join(save_path, img))
        print("\r"+"{}/{}".format(index+1, len(dirs)), end='', flush=True)
    print("Finished")

if __name__ == "__main__":
    copy2dir()
    # new_size = (256, 256)
    # root = r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface"

