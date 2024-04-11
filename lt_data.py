# -*- coding: utf-8 -*-
import os
from torchvision import transforms
from torch.utils.data import Dataset, random_split
from PIL import Image
import pickle


# 自定义数据集类
class UniqueImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = self.get_images()

    def get_images(self):
        images = []
        for filename in os.listdir(self.image_folder):
            image_path = os.path.join(self.image_folder, filename)
            if os.path.isfile(image_path):
                images.append(image_path)
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')  # 转换为灰度图
        if self.transform:
            image = self.transform(image)
        # 使用图像本身作为标签
        label = image

        return image, label


def datatset_pre(img_w, img_h, image_folder, train_dataset_path, val_dataset_path):
    # 定义图片转换流程：转换为img_w, img_h的灰度图，并归一化到[0, 1]范围
    transform = transforms.Compose([
        transforms.Resize((img_w, img_h)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # 创建自定义数据集实例
    dataset = UniqueImageDataset(image_folder, transform=transform)
    # 划分数据集为训练集和验证集，例如80%为训练集，20%为验证集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 保存训练集
    with open(train_dataset_path, 'wb') as f:
        pickle.dump(train_dataset, f)

    # 保存测试集
    with open(val_dataset_path, 'wb') as f:
        pickle.dump(test_dataset, f)
