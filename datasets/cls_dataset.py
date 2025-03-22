import torch
from torch.utils.data import Dataset
from torchvision import transforms as t
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

# 固定随机数种子
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def default_loader(path):
    return Image.open(path).convert('RGB')


class cls_dataset(Dataset):
    def __init__(self, root_dir, pathfile, transform=None, loader=default_loader, mode='clean'):
        self.imgs = []

        with open(pathfile, 'r') as pf:
            for line in pf:
                words = line.strip().split()
                if mode == 'clean':
                    img_path = root_dir + words[0]
                elif mode == 'adv':
                    img_path = root_dir + words[0].split('/')[-1].split('.')[0] + '_adv.png'
                else:
                    img_path = root_dir + words[0].split('.')[0] + '.png'
                label = int(words[1])
                name = words[0].split('/')[-1].split('.')[0]
                self.imgs.append((img_path, label, name))
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img, label, name = self.imgs[index]
        img = self.loader(img)
        if self.transform:
            img = self.transform(img)
        return img, label, name

    def __len__(self):
        return len(self.imgs)


def test_cls_dataset():
    root_dir = '../data/UCMerced_LandUse/'
    pathfile = '../data/UCM_test.txt'
    crop_size = 256

    test_augmentation = t.Compose([
        t.Resize((crop_size, crop_size)),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        t.RandomHorizontalFlip(),
        t.RandomVerticalFlip(),
        # t.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        # t.RandomErasing(),

    ])

    dataset = cls_dataset(root_dir, pathfile, transform=test_augmentation)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for batch_imgs, labels, batch_names in dataloader:
        # Convert labels to one-hot encoding
        num_classes = 45  # Example number of classes
        batch_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)

        # Visualize the batch
        plt.figure(figsize=(16, 16))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(batch_imgs[i].permute(1, 2, 0))
            plt.title(batch_names[i])
            plt.axis('off')



if __name__ == '__main__':
    test_cls_dataset()
