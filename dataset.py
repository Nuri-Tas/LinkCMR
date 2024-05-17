import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
from PIL import Image

class ACDC_Dataset(data.Dataset):
    def _init_(self, dataset_type='train', transform=None):
        self.dataset_type = dataset_type
        self.train_image = sorted(os.listdir("/YOUR-DATA-PATH/Dataset/train/Original"))
        self.train_gt = sorted(os.listdir("/YOUR-DATA-PATH/Dataset/train/GroundTruth"))

        self.val_image = sorted(os.listdir("/YOUR-DATA-PATH/Dataset/val/Original"))
        self.val_gt = sorted(os.listdir("/YOUR-DATA-PATH/Dataset/val/GroundTruth"))

        self.test_image = sorted(os.listdir("/YOUR-DATA-PATH/Dataset/test/Original"))
        self.test_gt = sorted(os.listdir("/YOUR-DATA-PATH/Dataset/test/GroundTruth"))

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

    def _getitem_(self, index):
        img_name = self.images[index]
        label_name = self.labels[index]
        image = Image.open(f"/YOUR-DATA-PATH/Dataset/{self.dataset_type}/Original/" + img_name).convert("RGB")
        label = Image.open(f"/YOUR-DATA-PATH/Dataset/{self.dataset_type}/GroundTruth/" + label_name).convert("L")
        label = np.array(label)
        mask = Image.fromarray(np.uint8(label))

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask

    def _len_(self):
        return len(self.images)