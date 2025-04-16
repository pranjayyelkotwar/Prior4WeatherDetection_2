import torch
from torch.utils.data import DataLoader
from models.detection_network import DomainAdaptiveFasterRCNN
from utils.data.cityscapes_clean_dataset import Cityscapes_Clean_Dataset, cityscapes_clean_dataset_collate_fn
from utils.data.cityscapes_foggy_dataset import Cityscapes_Foggy_Dataset, cityscapes_foggy_dataset_collate_fn

def test_forward_pass():
    # Initialize model
    model = DomainAdaptiveFasterRCNN(num_classes=10, backbone_name='vgg16', verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create dummy data for source and target datasets
    source_images = torch.rand(2, 3, 600, 800).to(device)  # Batch of 2 RGB images
    source_prior_images = torch.rand(2, 3, 600, 800).to(device)  # Batch of 2 prior images
    source_targets = [
        {
            'boxes': torch.tensor([[100, 150, 200, 250], [300, 350, 400, 450]], dtype=torch.float32).to(device),
            'labels': torch.tensor([1, 2], dtype=torch.int64).to(device),
            'image_id': torch.tensor([0]).to(device),
            'area': torch.tensor([5000, 10000], dtype=torch.float32).to(device),
            'iscrowd': torch.tensor([0, 0], dtype=torch.int64).to(device),
            'is_source': True
        },
        {
            'boxes': torch.tensor([[120, 170, 220, 270], [320, 370, 420, 470]], dtype=torch.float32).to(device),
            'labels': torch.tensor([3, 4], dtype=torch.int64).to(device),
            'image_id': torch.tensor([1]).to(device),
            'area': torch.tensor([5500, 10500], dtype=torch.float32).to(device),
            'iscrowd': torch.tensor([0, 0], dtype=torch.int64).to(device),
            'is_source': True
        }
    ]

    target_images = torch.rand(2, 3, 600, 800).to(device)  # Batch of 2 RGB images
    target_prior_images = torch.rand(2, 3, 600, 800).to(device)  # Batch of 2 prior images
    target_targets = [
        {
            'boxes': torch.empty((0, 4), dtype=torch.float32).to(device),
            'labels': torch.empty((0,), dtype=torch.int64).to(device),
            'image_id': torch.tensor([2]).to(device),
            'area': torch.empty((0,), dtype=torch.float32).to(device),
            'iscrowd': torch.empty((0,), dtype=torch.int64).to(device),
            'is_source': False
        },
        {
            'boxes': torch.empty((0, 4), dtype=torch.float32).to(device),
            'labels': torch.empty((0,), dtype=torch.int64).to(device),
            'image_id': torch.tensor([3]).to(device),
            'area': torch.empty((0,), dtype=torch.float32).to(device),
            'iscrowd': torch.empty((0,), dtype=torch.int64).to(device),
            'is_source': False
        }
    ]

    # Perform forward pass for source domain
    model.train()
    source_losses = model(source_images, source_prior_images, source_targets)
    print("Source domain losses:", source_losses)

    # Perform forward pass for target domain
    target_losses = model(target_images, target_prior_images, target_targets)
    print("Target domain losses:", target_losses)

if __name__ == "__main__":
    test_forward_pass()