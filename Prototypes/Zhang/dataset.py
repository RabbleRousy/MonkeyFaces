import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

class Monkey_Faces(Dataset):
    """
    Almost the same with sample code.
    Differences:
    >>> Use cv2 to read images
    >>> Build id_class mapping
    >>> Different pre-processing methods (torchvision.transforms)
    """
    def __init__(self, dataset_path, class_id_mapping, transform=None, dataset_type="train"):
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.class_id_mapping = class_id_mapping
        self.true_class_id = {}

        for label, dir_name in enumerate(sorted(os.listdir(dataset_path))):
            indi_dir = os.path.join(dataset_path, dir_name)
            for img_name in os.listdir(indi_dir):
                if ".simDB" not in img_name:
                    if dataset_type=="train":
                        self.labels.append(label)
                    elif dataset_type=="test":
                        if dir_name not in self.class_id_mapping:           # skip unseen category
                            continue
                        self.labels.append(self.class_id_mapping[dir_name])
                    else:
                        raise ValueError("Wrong dataset type. Please type again.")
                    self.img_paths.append(os.path.join(indi_dir, img_name))
            if dataset_type == "train":
                if dir_name not in self.class_id_mapping:
                    self.class_id_mapping[dir_name] = label
            if dataset_type != "train":
                if dir_name not in self.true_class_id:
                    self.true_class_id[dir_name] = self.class_id_mapping[dir_name]
        if len(self.labels) != len(self.img_paths):     # labels and images should have the same number
            print("Length: label: {} | images: {}".format(len(self.labels), len(self.img_paths)))
            raise ValueError("# of labels != # of images, please check the dataset")
                    

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.img_paths) 
    

def load_dataset(args):
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
    val_dataset = []
    if args.val:
        val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [len(test_dataset)-int(len(test_dataset)*0.5), int(len(test_dataset)*0.5)])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) if args.val else []
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False) if args.test else []
                                            
    print("train set length: {} | val set length: {} | test set length: {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))

    return train_loader, val_loader, test_loader

def count_images_num(root):
    dirs = sorted(os.listdir(root)) 
    count = {}
    dirname_index = {}
    for index, dir in enumerate(dirs):
        count[dir] = len(os.listdir(os.path.join(root, dir)))
        temp_dir = dir.split("'")[0]
        if temp_dir not in dirname_index:
            dirname_index[temp_dir] = index
    return count, dirname_index

if __name__ == "__main__":
    root = r"E:\datasets\monkeys\facedata_yamada\facedata_yamada\train_Magface"
    count, _ = count_images_num(root)
    print(count)