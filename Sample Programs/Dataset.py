# ファイル数が多いためその都度ptファイルを読みだようにする。データセットクラスの実装の一例
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.class_to_idx = {}

        # クラス名とインデックスの対応を作成
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.pt'):
                        self.image_files.append(os.path.join(class_dir, file_name))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_tensor = torch.load(img_path)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        label = self.labels[idx]
        return img_tensor, label