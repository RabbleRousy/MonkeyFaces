import os
import cv2
from torch.utils.data import Dataset, DataLoader

class Monkey_Faces(Dataset):
    def __init__(self, dataset_path, id_class_mapping, transform=None):
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.id_class_mapping = id_class_mapping

        for label, dir_name in enumerate(sorted(os.listdir(dataset_path))):
            indi_dir = os.path.join(dataset_path, dir_name)
            for img_name in os.listdir(indi_dir):
                if ".simDB" not in img_name:
                    self.img_paths.append(os.path.join(indi_dir, img_name))
                    self.labels.append(label)
            if label not in self.id_class_mapping:
                self.id_class_mapping[label] = dir_name

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