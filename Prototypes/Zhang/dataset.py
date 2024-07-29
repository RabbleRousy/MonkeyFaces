import os
import cv2
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
                    

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.img_paths) 

if __name__ == "__main__":
    root_dir = r"E:\datasets\monkeys\demo"
    dataset = Monkey_Faces(root_dir)
    dataset_loader = DataLoader(dataset)
    count = 0
    for item in dataset_loader:
        print(item)
        count += 1
        if count ==5:
            break