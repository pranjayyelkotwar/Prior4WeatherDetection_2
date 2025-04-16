import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2  # Required for OpenCV functions

# from Prior4WeatherDetection.utils.prior_calculation.dark_channel_prior_estimation_utils import estimate_transmission_map, refine_transmission
from utils.prior_calculation.dark_channel_prior_estimation_utils import estimate_transmission_map, refine_transmission

class Cityscapes_Foggy_Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, beta_levels=[0.01, 0.02, 0.05]):
        """
        Args:
            root_dir (str): Root directory of the Cityscapes Foggy dataset.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied to images.
            beta_levels (list): List of beta levels to include (default: [0.01, 0.02, 0.05])
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.beta_levels = beta_levels

        # Paths to images and annotations
        self.img_dir = os.path.join(root_dir, 'leftImg8bit_foggy', split)
        self.ann_dir = os.path.join(root_dir, 'gtFine', split)  # Same annotations as clean
        
        self.valid_labels = {
            'person': 0, 'rider': 1, 'car': 2, 'truck': 3,
            'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7
        }

        self.id_to_label = {v: k for k, v in self.valid_labels.items()}

        # Collect all image files with available foggy versions
        self.img_files = []
        for city in os.listdir(self.img_dir):
            city_img_dir = os.path.join(self.img_dir, city)
            for img_file in os.listdir(city_img_dir):
                # Check if it's one of the foggy versions we want
                for beta in self.beta_levels:
                    if img_file.endswith(f'_leftImg8bit_foggy_beta_{beta:.2f}.png'):
                        # Extract base filename to match with clean version
                        base_name = img_file.replace(f'_leftImg8bit_foggy_beta_{beta:.2f}.png', '_leftImg8bit.png')
                        self.img_files.append((city, img_file, base_name, beta))
                        break  # Only need to add once per base image

        self.img_files.sort()  # Ensure consistent order
        print(f"Found {len(self.img_files)} foggy images in {split} split for beta levels {beta_levels}.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        city, img_file, base_name, beta = self.img_files[idx]

        # Load foggy image
        img_path = os.path.join(self.img_dir, city, img_file)
        img = Image.open(img_path).convert('RGB')

        # Resize image (shorter side to 600)
        w, h = img.size
        scale = 600 / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        resize_transform = transforms.Resize((new_h, new_w))
        img = resize_transform(img)

        # Convert PIL image to NumPy array for prior computation
        img_np = np.array(img)

        # Generate prior image using your transmission map functions
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        transmission, _ = estimate_transmission_map(img_bgr, window_size=15, omega=0.95, t0=0.1)
        prior_np = refine_transmission(transmission, img_bgr)

        # Placeholder for transmission map functions (define these functions or replace with actual implementations)
        transmission = np.ones_like(img_bgr[:, :, 0], dtype=np.float32)  # Dummy transmission map
        prior_np = transmission  # Dummy refinement
        prior_img = Image.fromarray((prior_np * 255).astype(np.uint8))
        prior_img = resize_transform(prior_img)

        # Load annotations (using base name to match clean version)
        ann_file = base_name.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        ann_path = os.path.join(self.ann_dir, city, ann_file)

        boxes, labels = [], []
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)

            for obj in ann_data['objects']:
                label = obj['label']
                if label in self.valid_labels:
                    polygon = np.array(obj['polygon'])
                    x_min, y_min = polygon.min(axis=0)
                    x_max, y_max = polygon.max(axis=0)

                    # Scale bounding box coordinates
                    x_min, x_max = x_min * scale, x_max * scale
                    y_min, y_max = y_min * scale, y_max * scale

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(self.valid_labels[label])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.size(0) > 0 else torch.tensor([])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        # Apply transforms
        if self.transform:
            img = self.transform(img)
            prior_img = self.transform(prior_img)
        else:
            img = transforms.ToTensor()(img)
            prior_img = transforms.ToTensor()(prior_img)

        # Target dictionary
        target = {
            'is_source' : 0,
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
            'beta': torch.tensor(beta, dtype=torch.float32)  # Convert beta to tensor
        }

        return img, prior_img, target


# Add a collate function for the foggy dataset
def cityscapes_foggy_dataset_collate_fn(batch):
    images, prior_images, targets = zip(*batch)
    return list(images), list(prior_images), list(targets)