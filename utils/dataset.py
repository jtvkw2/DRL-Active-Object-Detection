import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

class COCODataset(Dataset):
    """
    A class for handling the COCO dataset, including loading the data, preprocessing, and data augmentation.
    """

    def __init__(self, data_dir, set_name='train2017', transform=None):
        """
        Initializes the COCODataset class.

        Args:
            data_dir (str): The path to the dataset directory.
            set_name (str): The name of the dataset split ('train2017', 'val2017', or 'test2017').
            transform (callable, optional): A function/transform to apply to the input images.
        """
        self.data_dir = data_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(data_dir, 'annotations',
                         f'instances_{set_name}.json'))
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Returns an image and its corresponding target annotations.

        Args:
            index (int): The index of the sample in the dataset.

        Returns:
            image (PIL.Image.Image): The input image.
            target (dict): A dictionary containing the target annotations.
        """
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        image_path = os.path.join(
            self.data_dir, self.set_name, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=self.image_ids[index])
        anns = self.coco.loadAnns(ann_ids)
        boxes = np.array([ann['bbox'] for ann in anns], dtype=np.float32)

        # Convert [x, y, w, h] format to [x_min, y_min, x_max, y_max] format
        boxes[:, 2:] += boxes[:, :2]

        labels = np.array([ann['category_id'] for ann in anns], dtype=np.int64)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def collate_fn(self, batch):
        """
        A custom collate function for the DataLoader.

        Args:
            batch (list): A list of (image, target) tuples from the dataset.

        Returns:
            images (torch.Tensor): A tensor containing the batched images.
            targets (list): A list of dictionaries containing the batched target annotations.
        """
        images, targets = zip(*batch)
        images = torch.stack([torch.tensor(
            np.array(img), dtype=torch.float32).permute(2, 0, 1) for img in images])

        return images, targets
