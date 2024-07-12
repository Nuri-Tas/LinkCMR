import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
from PIL import Image

class ACDC_Dataset(data.Dataset):
    def __init__(self, dataset_type='train', dataset_name="ACDC", transform=None):
        self.dataset_type = dataset_type
        self.dataset_name = dataset_name
        self.train_image = sorted(os.listdir(f"Dataset/{dataset_name}/train/Original"))
        self.train_gt = sorted(os.listdir(f"Dataset/{dataset_name}/train/GroundTruth"))

        self.val_image = sorted(os.listdir(f"Dataset/{dataset_name}/val/Original"))
        self.val_gt = sorted(os.listdir(f"Dataset/{dataset_name}/val/GroundTruth"))

        self.test_image = sorted(os.listdir(f"Dataset/{dataset_name}/test/Original"))
        self.test_gt = sorted(os.listdir(f"Dataset/{dataset_name}/test/GroundTruth"))

        self.transform = transform

        if dataset_type=='train':
            self.images = self.train_image
            self.labels = self.train_gt
        elif dataset_type=='val':
            self.images = self.val_image
            self.labels = self.val_gt
        elif dataset_type=='test':
            self.images = self.test_image
            self.labels = self.test_gt

    def __getitem__(self, index):
        img_name = self.images[index]
        label_name = self.labels[index]
        image = Image.open(f"Dataset/{self.dataset_name}/{self.dataset_type}/Original/" + img_name).convert("RGB")
        label = Image.open(f"Dataset/{self.dataset_name}/{self.dataset_type}/GroundTruth/" + label_name).convert("L")
        label = np.array(label)
        mask = Image.fromarray(np.uint8(label))

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask

    def __len__(self):
        return len(self.images)


class CVCClinicDB_Dataset(data.Dataset):
    def __init__(self, dataset_type='train', transform=None):
        self.item_image = sorted(os.listdir("Dataset/CVC_Clinical_DB/Original"))
        self.item_gt = sorted(os.listdir("Dataset/CVC_Clinical_DB/Ground Truth"))

        self.transform = transform

        if dataset_type=='train':
            self.images = self.item_image[:368]
            self.labels = self.item_gt[:368]
        elif dataset_type=='val':
            self.images = self.item_image[368:490]
            self.labels = self.item_gt[368:490]
        elif dataset_type=='test':
            self.images = self.item_image[490:]
            self.labels = self.item_gt[490:]

    def __getitem__(self, index):
        img_name = self.images[index]
        label_name = self.labels[index]
        image = Image.open("Dataset/CVC_Clinical_DB/Original/" + img_name).convert("RGB")
        label = Image.open("Dataset/CVC_Clinical_DB/Ground Truth/" + label_name).convert("L")
        label = np.array(label)
        mask = np.where(label>200, 1, 0)
        mask = Image.fromarray(np.uint8(mask))

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask

    def __len__(self):
        return len(self.images)
