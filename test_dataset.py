import torch
import torch.optim as optim
import torch.nn.utils as utils  # Added for gradient clipping
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np

batch_size = 1

from utils.data.cityscapes_clean_dataset import Cityscapes_Clean_Dataset, cityscapes_clean_dataset_collate_fn
from utils.data.cityscapes_foggy_dataset import Cityscapes_Foggy_Dataset, cityscapes_foggy_dataset_collate_fn

# Create output directory for visualizations
output_dir = "visualization_output"
os.makedirs(output_dir, exist_ok=True)

def save_visualization(images, prior_images, targets, dataset, num_images=2, output_path='visualization.png'):
    """
    Visualize a batch of images with their prior images side by side, including bounding boxes and labels,
    and save the visualization to a file.
    
    Args:
        images (list): List of image tensors [C, H, W].
        prior_images (list): List of prior image tensors [C, H, W].
        targets (list): List of target dictionaries with 'boxes' and 'labels'.
        dataset: Dataset instance to get label mapping.
        num_images (int): Number of images to visualize from the batch.
        output_path (str): Path to save the visualization.
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
            label_name = dataset.id_to_label[label.item()]
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
            label_name = dataset.id_to_label[label.item()]
            axes[1, i].text(x_min, y_min - 5, label_name, color='white', fontsize=10,
                         bbox=dict(facecolor='red', alpha=0.5))
    
    plt.tight_layout()
    # Save the figure instead of displaying it
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to {output_path}")

# Dataset paths
data_root = "/teamspace/studios/this_studio/Prior4WeatherDetection/datataset/cityscapes"

# Training datasets
source_dataset = Cityscapes_Foggy_Dataset(
    root_dir=data_root,
    split='train',
    transform=None
)

# Dataloaders with smaller batch size for stability
source_loader = DataLoader(
    source_dataset,
    batch_size = batch_size,
    shuffle=True,
    collate_fn=cityscapes_foggy_dataset_collate_fn,
    num_workers=4
)

# Visualize multiple batches
num_batches_to_visualize = 4
for i, batch in enumerate(source_loader):
    if i >= num_batches_to_visualize:
        break
        
    images, prior_images, targets = batch
    
    # Save visualization
    save_visualization(
        images, 
        prior_images, 
        targets, 
        source_dataset,
        num_images=len(images), 
        output_path=os.path.join(output_dir, f"batch_{i+1}_visualization.png")
    )
    
    print(f"Processed batch {i+1}/{num_batches_to_visualize}")
    
print(f"All visualizations saved to {output_dir}")