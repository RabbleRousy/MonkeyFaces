# 前処理の一例
import os
from PIL import Image
from PIL import ImageOps
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, resize=(227, 227)):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor()
    ])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.endswith(('jpg', 'jpeg', 'png')):
                file_path = os.path.join(root, file)
                img = Image.open(file_path).convert('RGB')

                # 縦横比を維持しながらリサイズする
                img.thumbnail(resize, Image.ANTIALIAS)

                # パディングを追加して227x227にする
                delta_w = resize[0] - img.size[0]
                delta_h = resize[1] - img.size[1]
                padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
                img = ImageOps.expand(img, padding, fill=(0, 0, 0))

                img_tensor = transform(img)

                # 保存先のディレクトリを作成
                relative_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, relative_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_path = os.path.join(save_dir, file.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt'))
                torch.save(img_tensor, save_path)