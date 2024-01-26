import os
import pandas as pd
import numpy as np
import PIL
import torch
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as F


class HSCGalaxyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_ids = dataframe['object_id'].unique()

    def __getitem__(self, idx):
        # load images and masks
        image_id = self.image_ids[idx]
        records = self.df[self.df['object_id'] == image_id]

        img_path = os.path.join(self.image_dir, 'pngs', str(image_id) + '.png')
        mask_path = os.path.join(self.image_dir, 'masks', str(image_id) + '_mask.png')
        img = read_image(img_path)
        mask = read_image(mask_path)

        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask and handle empty bounding boxes
        # boxes = masks_to_boxes(masks)
        boxes = records[['bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax']].values
        if np.isnan(boxes).all():
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes,dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        labels_list = records[['labels']].values
        labels = torch.as_tensor(labels_list, dtype=torch.int64)
        labels = torch.squeeze(labels, 1)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target['boxes'] = tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=F.get_size(img))
        target['masks'] = tv_tensors.Mask(masks)
        target['labels'] = labels
        target['image_id'] = torch.tensor([image_id])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_ids)


class FRCNNGalaxyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, image_dir, band=None, colour=False, transforms=None):
        # super().__init__()
        self.df = dataframe
        self.image_dir = image_dir
        self.band = band
        self.colour = colour
        self.transforms = transforms
        self.image_ids = dataframe['local_ids'].unique()
        
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['local_ids'] == image_id]
        file_name = str(records['local_ids'].iloc[0]) + '_{}.png'.format(self.band)

        if not self.colour:
            image = PIL.Image.open(f"{self.image_dir}{file_name}").convert('L')
        else:
            image = PIL.Image.open(f"{self.image_dir}{file_name}").convert('RGB')
        
        image = tv_tensors.Image(image)
        
        # Handle empty bounding boxes
        boxes = records[['x1', 'y1', 'x2', 'y2']].values
        if np.isnan(boxes).all():
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes,dtype=torch.float32)
        
        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=image.shape[-2:]
        )
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        
        # there is only one class supported
        # labels = torch.ones((records.shape[0], ), dtype=torch.int64)
        labels_list = records[['label']].values
        labels = torch.as_tensor(labels_list, dtype=torch.int64)
        labels = torch.squeeze(labels, 1)
        
        # no crowd instances
        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([image_id])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]