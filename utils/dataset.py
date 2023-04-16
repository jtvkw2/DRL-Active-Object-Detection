from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import cv2
import numpy as np
from config.config import config
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, root_dir, set_name, image_dir, annotation_path, transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = Compose([
            Resize((config['img_size'],
                   config['img_size'])),
            ToTensor()
        ])

        self._image_dir = image_dir
        self._annotation_path = annotation_path

        self.coco = COCO(self._annotation_path)
        self.data = self.load_data()

    def load_data(self):
        data = []
        for image_id in self.coco.getImgIds():
            data.append(self.coco.loadImgs(image_id)[0])
        return data

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = os.path.join(self._image_dir, img_data['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array to a PIL image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.data)
