import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.detection_network import DomainAdaptiveFasterRCNN
from utils.data.cityscapes_clean_dataset import Cityscapes_Clean_Dataset
from utils.data.cityscapes_foggy_dataset import Cityscapes_Foggy_Dataset


def detection_collate_fn(batch):
    return tuple(zip(*batch))

# Initialize model
model = DomainAdaptiveFasterRCNN(num_classes=10, backbone_name='vgg16')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer (SGD as in paper)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

# Datasets
source_dataset = Cityscapes_Clean_Dataset("/teamspace/studios/this_studio/Prior4WeatherDetection/datataset/cityscapes")  # Clean images + labels
target_dataset = Cityscapes_Foggy_Dataset("/teamspace/studios/this_studio/Prior4WeatherDetection/datataset/cityscapes")  # Hazy/Rainy images + priors

# Dataloaders
source_loader = DataLoader(
    source_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=detection_collate_fn
)

target_loader = DataLoader(
    target_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=detection_collate_fn
)
# Training loop
for epoch in range(70):  # 70 epochs as in paper
    model.train()
    
    # Alternate between source and target batches
    for source_batch, target_batch in zip(source_loader, target_loader):
        source_images, source_priors, source_targets = source_batch
        target_images, target_priors, target_targets = target_batch

        # --- Source Domain Training ---
        source_images = torch.stack(source_images).to(device)
        source_targets = [
            {**{k: v.to(device) for k, v in t.items()}, 'is_source': True}
            for t in source_targets
        ]
        
        # Forward pass (compute detection loss)
        losses = model(source_images, source_targets)
        
        # Backward pass (only detection loss)
        loss = losses['loss_classifier'] + losses['loss_box_reg'] + losses['loss_objectness'] + losses['loss_rpn_box_reg']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # --- Target Domain Training ---
        target_images = torch.stack(target_images).to(device)
        target_priors = torch.stack(target_priors).to(device)
        target_targets = [
            {'prior': target_priors[i], 'is_source': False}
            for i in range(len(target_images))
        ]
        
        # Forward pass (compute PAL + RFRB regularization)
        losses = model(target_images, target_targets)
        
        # Backward pass (PAL + regularization)
        loss = losses['loss_pal'] + losses['loss_reg']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Adjust learning rate (after 50K iterations, reduce to 0.0001)
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    
    print(f"Epoch {epoch}, Loss: {loss.item()}")