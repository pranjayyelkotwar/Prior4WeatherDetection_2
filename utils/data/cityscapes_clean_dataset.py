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

#from Prior4WeatherDetection.utils.prior_calculation.dark_channel_prior_estimation_utils import estimate_transmission_map, refine_transmission
from utils.prior_calculation.dark_channel_prior_estimation_utils import estimate_transmission_map, refine_transmission

class Cityscapes_Clean_Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the Cityscapes dataset.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Paths to images and annotations
        self.img_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.ann_dir = os.path.join(root_dir, 'gtFine', split)
        
        self.valid_labels = {
            'person': 0, 'rider': 1, 'car': 2, 'truck': 3,
            'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7
        }

        self.id_to_label = {v: k for k, v in self.valid_labels.items()}

        # Collect all image files
        self.img_files = []
        for city in os.listdir(self.img_dir):
            city_img_dir = os.path.join(self.img_dir, city)
            for img_file in os.listdir(city_img_dir):
                if img_file.endswith('_leftImg8bit.png'):
                    self.img_files.append((city, img_file))

        self.img_files.sort()  # Ensure consistent order
        print(f"Found {len(self.img_files)} images in {split} split.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        city, img_file = self.img_files[idx]

        # Load main image
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

        # Load annotations
        ann_file = img_file.replace('_leftImg8bit.png', '_gtFine_polygons.json')
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
            'is_source' : 1,
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        return img, prior_img, target


# Modified collate function
def cityscapes_clean_dataset_collate_fn(batch):
    images, prior_images, targets = zip(*batch)
    return list(images), list(prior_images), list(targets)

def visualize_batch(images, prior_images, targets, num_images=2):
    """
    Visualize a batch of images with their prior images side by side, including bounding boxes and labels.
    
    Args:
        images (list): List of image tensors [C, H, W].
        prior_images (list): List of prior image tensors [C, H, W].
        targets (list): List of target dictionaries with 'boxes' and 'labels'.
        num_images (int): Number of images to visualize from the batch.
    """
    # Create a grid with 2 rows and num_images columns
    fig, axes = plt.subplots(2, num_images, figsize=(15, 10))
    
    # Ensure axes is always a 2D array
    if num_images == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(num_images, len(images))):
        # Convert tensor to numpy for plotting (H, W, C)
        img = images[i].permute(1, 2, 0).numpy()
        prior_img = prior_images[i].permute(1, 2, 0).numpy()
        
        # If prior image is single channel, convert to 3 channels for display
        if prior_img.shape[2] == 1:
            prior_img = np.repeat(prior_img, 3, axis=2)
            
        boxes = targets[i]['boxes'].numpy()
        labels = targets[i]['labels'].numpy()
        
        # Plot original image (top row)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Original Image {i+1}")
        
        # Add bounding boxes and labels to original image
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none'
            )
            axes[0, i].add_patch(rect)
            
            # Add label text
            label_name = dataset.id_to_label[label]
            axes[0, i].text(x_min, y_min - 5, label_name, color='white', fontsize=10,
                         bbox=dict(facecolor='red', alpha=0.5))
        
        # Plot prior image (bottom row)
        axes[1, i].imshow(prior_img)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Prior Image {i+1}")
        
        # Add the same bounding boxes to the prior image
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none'
            )
            axes[1, i].add_patch(rect)
            
            # Add label text
            label_name = dataset.id_to_label[label]
            axes[1, i].text(x_min, y_min - 5, label_name, color='white', fontsize=10,
                         bbox=dict(facecolor='red', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


# # Example usage
# if __name__ == "__main__":
#     # Define dataset and dataloader
#     dataset = CityscapesDataset(root_dir='/teamspace/studios/this_studio/Prior4WeatherDetection/datataset/cityscapes', split='train')
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
#     # Get and visualize one batch
#     data_iterator = iter(dataloader)
#     images, prior_images, targets = next(data_iterator)
    
#     images, prior_images, targets = next(data_iterator)
#     print(f"Image shapes: {[img.shape for img in images]}")
#     print(f"Prior image shapes: {[prior.shape for prior in prior_images]}")
#     print(f"Target box shapes: {[t['boxes'].shape for t in targets]}")
    
#     visualize_batch(images, prior_images, targets, num_images=2)