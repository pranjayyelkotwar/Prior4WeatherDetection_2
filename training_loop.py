import torch
import torch.optim as optim
import torch.nn.utils as utils  # Added for gradient clipping
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
import os
import time
import math  # Added for isfinite check
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast  # Add these imports for mixed precision
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # Add cosine scheduler

from models.detection_network import DomainAdaptiveFasterRCNN
from utils.data.cityscapes_clean_dataset import Cityscapes_Clean_Dataset, cityscapes_clean_dataset_collate_fn
from utils.data.cityscapes_foggy_dataset import Cityscapes_Foggy_Dataset, cityscapes_foggy_dataset_collate_fn
from utils.evaluation import evaluate_detection_streaming
import gc

batch_size = 16
num_epochs = 25
# Configure wandb with more detailed configuration
wandb.init(
    project="prior4weatherdetection", 
    name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    config={
        "num_epochs": num_epochs,
        "learning_rate": 0.0005,
        "lr_decay_epoch": 30,
        "lr_decay_factor": 0.1,
        "batch_size": batch_size,
        "backbone": "vgg16",
        "num_classes": 8
    }
)

# Custom scheduler that combines linear warmup with cosine annealing with warm restarts
class LinearWarmupCosineAnnealingScheduler:
    """Custom scheduler that combines linear warmup with cosine annealing with warm restarts"""
    def __init__(self, optimizer, warmup_epochs, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step(0)  # Initialize
        
    def get_lr(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            return [lr * (epoch + 1) / self.warmup_epochs for lr in self.base_lrs]
        else:
            # Cosine annealing with warm restarts
            actual_epoch = epoch - self.warmup_epochs
            if self.T_mult == 1:
                cycle = actual_epoch // self.T_0
                cycle_epoch = actual_epoch % self.T_0
                cos_factor = 0.5 * (1 + np.cos(np.pi * cycle_epoch / self.T_0))
            else:
                n = 0
                cycle_epoch = actual_epoch
                while cycle_epoch >= self.T_0 * (self.T_mult ** n):
                    cycle_epoch -= self.T_0 * (self.T_mult ** n)
                    n += 1
                cycle = n
                cos_factor = 0.5 * (1 + np.cos(np.pi * cycle_epoch / (self.T_0 * (self.T_mult ** (cycle)))))
            
            return [self.eta_min + (lr - self.eta_min) * cos_factor for lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
            values = self.get_lr(self.last_epoch)
        else:
            self.last_epoch = epoch
            values = self.get_lr(epoch)
            
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        
        return values

# Initialize model
model = DomainAdaptiveFasterRCNN(num_classes=8, backbone_name='vgg16', verbose=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Log model architecture to wandb
wandb.watch(model, log="all", log_freq=10)

# Increase initial learning rate for faster convergence
initial_lr = 0.003  # Increased further from 0.001 to help break through plateaus
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0005)

# Use our custom scheduler with warmup to address plateauing
scheduler = LinearWarmupCosineAnnealingScheduler(
    optimizer,
    warmup_epochs=2,     # First 2 epochs are for warmup
    T_0=4,               # Restart every 4 epochs after warmup
    T_mult=2,            # Double the restart period after each restart
    eta_min=1e-6         # Minimum learning rate
)

# Dataset paths
data_root = "/teamspace/studios/this_studio/Prior4WeatherDetection/datataset/cityscapes"

# Training datasets
source_dataset = Cityscapes_Clean_Dataset(
    root_dir=data_root,
    split='train',
    transform=None
)

target_dataset = Cityscapes_Foggy_Dataset(
    root_dir=data_root,
    split='train',
    transform=None
)

# Validation datasets
val_source_dataset = Cityscapes_Clean_Dataset(
    root_dir=data_root,
    split='val',
    transform=None
)

val_target_dataset = Cityscapes_Foggy_Dataset(
    root_dir=data_root,
    split='val',
    transform=None
)

# Dataloaders with optimized settings for maximum performance
source_loader = DataLoader(
    source_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=cityscapes_clean_dataset_collate_fn,
    num_workers=8,  # Increased from 4 to 8 for faster data loading
    pin_memory=True,  # Enable pin_memory for faster CPU->GPU transfers
    prefetch_factor=2,  # Prefetch 2 batches ahead
    persistent_workers=True  # Keep workers alive between batches
)

target_loader = DataLoader(
    target_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=cityscapes_foggy_dataset_collate_fn,
    num_workers=8,  # Increased from 4 to 8 for faster data loading
    pin_memory=True,  # Enable pin_memory for faster CPU->GPU transfers
    prefetch_factor=2,  # Prefetch 2 batches ahead
    persistent_workers=True  # Keep workers alive between batches
)

# Use smaller batch size for validation to avoid OOM issues
val_batch_size = batch_size // 2

val_source_loader = DataLoader(
    val_source_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    collate_fn=cityscapes_clean_dataset_collate_fn,
    num_workers=4,  # Use fewer workers for validation
    pin_memory=True
)

val_target_loader = DataLoader(
    val_target_dataset,
    batch_size = batch_size,
    shuffle=False,
    collate_fn=cityscapes_foggy_dataset_collate_fn,
    num_workers=4
)

# Config with more gradual learning rate schedule
config = {
    "num_epochs": num_epochs,
    "learning_rate": 0.0005,  # Lower initial learning rate
    "lr_decay_epoch": 30,     # Earlier decay
    "lr_decay_factor": 0.1,
    "save_interval": 5,
    "eval_interval": 2,
    "checkpoint_dir": "./checkpoints",
    "max_grad_norm": 5.0,     # Maximum gradient norm for clipping
}

# Create checkpoint directory if it doesn't exist
os.makedirs(config["checkpoint_dir"], exist_ok=True)

# Function to save checkpoint
def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, os.path.join(config["checkpoint_dir"], f"model_epoch_{epoch}.pth"))
    print(f"Checkpoint saved for epoch {epoch}")

# Function to validate loss value is not NaN or Inf
def is_valid_loss(loss_value):
    if isinstance(loss_value, torch.Tensor):
        return torch.isfinite(loss_value).all().item()
    return math.isfinite(loss_value)

# Training loop with validation and stability improvements
def train(model, source_loader, target_loader, val_source_loader, val_target_loader, optimizer, device, config):
    skipped_batches = 0
    total_batches = 0
    lambda_reg = config.get("lambda_reg", 1.0)  # Regularization weight
    
    # Add scaler for mixed precision training
    scaler = GradScaler()
    
    for epoch in range(config["num_epochs"]):
        # Training phase
        model.train()
        combined_epoch_loss = 0.0
        source_det_loss = 0.0  # For tracking L_det^src
        pal_src_loss = 0.0     # For tracking L_pal^src
        pal_tgt_loss = 0.0     # For tracking L_pal^tgt
        reg_loss = 0.0         # For tracking L_reg
        
        num_batches = 0
        epoch_skipped_batches = 0
        start_time = time.time()
        
        # Create iterator for target_loader to handle different dataset sizes
        target_iter = iter(target_loader)
        
        source_tqdm = tqdm(source_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        
        for batch_idx, source_batch in enumerate(source_tqdm):
            total_batches += 1
            
            try:
                # Get a batch from target dataset
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)
                
                # Process source domain data
                source_images, source_prior_images, source_targets = source_batch
                source_images = [img.to(device) for img in source_images]
                source_prior_images = [img.to(device) for img in source_prior_images]
                source_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in source_targets]

                source_images_tensor = torch.stack(source_images)
                source_prior_images_tensor = torch.stack(source_prior_images)

                # Forward pass for source domain with mixed precision
                with autocast():
                    source_losses = model(source_images_tensor, source_prior_images_tensor, source_targets)
                
                # Check for valid losses and replace NaN/Inf values
                for k in list(source_losses.keys()):
                    if not is_valid_loss(source_losses[k]):
                        print(f"Warning: Invalid {k} in source batch {batch_idx}: {source_losses[k]}")
                        source_losses[k] = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Process target domain data
                target_images, target_prior_images, target_targets = target_batch
                target_images = [img.to(device) for img in target_images]
                target_prior_images = [img.to(device) for img in target_prior_images]
                target_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_targets]

                target_images_tensor = torch.stack(target_images)
                target_prior_images_tensor = torch.stack(target_prior_images)

                # Forward pass for target domain with mixed precision
                with autocast():
                    target_losses = model(target_images_tensor, target_prior_images_tensor, target_targets)
                
                # Check for valid losses and replace NaN/Inf values
                for k in list(target_losses.keys()):
                    if not is_valid_loss(target_losses[k]):
                        print(f"Warning: Invalid {k} in target batch {batch_idx}: {target_losses[k]}")
                        target_losses[k] = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Calculate L_det^src (detection loss from source)
                l_det_src = source_losses.get('loss_classifier', torch.tensor(0.0, device=device)) + \
                           source_losses.get('loss_box_reg', torch.tensor(0.0, device=device)) + \
                           source_losses.get('loss_objectness', torch.tensor(0.0, device=device)) + \
                           source_losses.get('loss_rpn_box_reg', torch.tensor(0.0, device=device))
                
                # Calculate L_pal^src and L_pal^tgt (PAL losses)
                l_pal_src = source_losses.get('loss_pal', torch.tensor(0.0, device=device))
                l_pal_tgt = target_losses.get('loss_pal', torch.tensor(0.0, device=device))
                
                # Calculate L_adv (adversarial loss)
                l_adv = 0.5 * (l_pal_src + l_pal_tgt)
                
                # Calculate L_reg (regularization loss)
                l_reg = target_losses.get('loss_reg', torch.tensor(0.0, device=device))
                
                # Calculate combined loss: L_det^src - L_adv + λL_reg
                combined_loss = l_det_src + l_adv + lambda_reg * l_reg
                
                if not is_valid_loss(combined_loss):
                    raise ValueError(f"Invalid combined loss: {combined_loss}")
                
                # Backward pass and optimization with gradient clipping
                optimizer.zero_grad()
                scaler.scale(combined_loss).backward()
                # Apply gradient clipping to prevent exploding gradients
                scaler.unscale_(optimizer)
                utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                
                # Track losses for logging
                combined_epoch_loss += combined_loss.item()
                source_det_loss += l_det_src.item()
                pal_src_loss += l_pal_src.item()
                pal_tgt_loss += l_pal_tgt.item()
                reg_loss += l_reg.item()
                
                num_batches += 1
                
                # Log per-batch losses to wandb
                global_step = epoch * len(source_loader) + batch_idx
                wandb.log({
                    "batch/combined_loss": combined_loss.item(),
                    "batch/source_det_loss": l_det_src.item(),
                    "batch/adversarial_loss": l_adv.item(),
                    "batch/reg_loss": l_reg.item(),
                    "batch/pal_src_loss": l_pal_src.item(),
                    "batch/pal_tgt_loss": l_pal_tgt.item(),
                    "global_step": global_step,
                    "lr": optimizer.param_groups[0]['lr']  # Log current learning rate
                })
                
                # Update tqdm description
                source_tqdm.set_postfix({
                    'loss': f"{combined_loss.item():.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                    'skipped': epoch_skipped_batches
                })
                
            except Exception as e:
                epoch_skipped_batches += 1
                skipped_batches += 1
                print(f"Error in batch {batch_idx}: {str(e)}")
                print(f"Skipping this batch and continuing training...")
                continue
        
        # Step the learning rate scheduler after each epoch
        scheduler.step()
        
        # Calculate average losses
        if num_batches > 0:
            avg_combined_loss = combined_epoch_loss / num_batches
            avg_source_det_loss = source_det_loss / num_batches
            avg_pal_src_loss = pal_src_loss / num_batches
            avg_pal_tgt_loss = pal_tgt_loss / num_batches
            avg_adv_loss = 0.5 * (avg_pal_src_loss + avg_pal_tgt_loss)
            avg_reg_loss = reg_loss / num_batches
        else:
            print("Warning: No valid batches in epoch. Using zeros for metrics.")
            avg_combined_loss = avg_source_det_loss = avg_pal_src_loss = 0.0
            avg_pal_tgt_loss = avg_adv_loss = avg_reg_loss = 0.0
        
        # Clear GPU cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Print memory stats for debugging
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            print(f"  GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

        # Perform garbage collection
        gc.collect()
        
        epoch_time = time.time() - start_time
        skip_rate = epoch_skipped_batches / (num_batches + epoch_skipped_batches) if (num_batches + epoch_skipped_batches) > 0 else 0
        
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] Time: {epoch_time:.2f}s, Skip rate: {skip_rate:.2%}")
        print(f"  Combined Loss: {avg_combined_loss:.4f}")
        print(f"  Components: L_det_src: {avg_source_det_loss:.4f}, L_adv: {avg_adv_loss:.4f}, L_reg: {avg_reg_loss:.4f}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/combined_loss": avg_combined_loss if is_valid_loss(avg_combined_loss) else 0.0,
            "train/source_det_loss": avg_source_det_loss if is_valid_loss(avg_source_det_loss) else 0.0,
            "train/adversarial_loss": avg_adv_loss if is_valid_loss(avg_adv_loss) else 0.0,
            "train/pal_src_loss": avg_pal_src_loss if is_valid_loss(avg_pal_src_loss) else 0.0,
            "train/pal_tgt_loss": avg_pal_tgt_loss if is_valid_loss(avg_pal_tgt_loss) else 0.0,
            "train/reg_loss": avg_reg_loss if is_valid_loss(avg_reg_loss) else 0.0,
            "train/epoch_time": epoch_time,
            "train/skip_rate": skip_rate,
            "lr": optimizer.param_groups[0]['lr']
        })

        # Validation phase
        if (epoch + 1) % config["eval_interval"] == 0:
            print("Running validation...")
            try:
                # Evaluate on source validation set
                source_metrics = evaluate_detection_streaming(
                    model, val_source_loader, device, num_classes=8
                )
                
                # Evaluate on target validation set
                target_metrics = evaluate_detection_streaming(
                    model, val_target_loader, device, num_classes=8
                )
                
                print('calculated target metrics')
                source_mAP = source_metrics['mAP']
                target_mAP = target_metrics['mAP']
                
                print(f"  Validation: Source mAP: {source_mAP:.4f}, Target mAP: {target_mAP:.4f}")
                
                # Log per-class AP to wandb
                source_ap_per_class = source_metrics['AP_per_class']
                target_ap_per_class = target_metrics['AP_per_class']
                
                for class_id, ap in source_ap_per_class.items():
                    wandb.log({f"val_source/AP_class_{class_id}": ap, "epoch": epoch + 1})
                
                for class_id, ap in target_ap_per_class.items():
                    wandb.log({f"val_target/AP_class_{class_id}": ap, "epoch": epoch + 1})
                
                # Log overall metrics
                wandb.log({
                    "val_source/mAP": source_mAP,
                    "val_target/mAP": target_mAP,
                    "epoch": epoch + 1
                })
            except Exception as e:
                print(f"Error during validation: {e}")
                print("Skipping this validation step and continuing training...")
        
        # Save checkpoint
        if (epoch + 1) % config["save_interval"] == 0:
            try:
                # Use the combined loss instead of sum of source and target losses
                save_checkpoint(epoch + 1, model, optimizer, avg_combined_loss)
                # Also save model to wandb
                wandb.save(os.path.join(config["checkpoint_dir"], f"model_epoch_{epoch+1}.pth"))
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

        # Learning rate schedule - more gradual decay
        if (epoch + 1) % config["lr_decay_epoch"] == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * config["lr_decay_factor"]
                print(f"Adjusted learning rate to {param_group['lr']}")
        
        # Add early stopping if skip rate is too high
        if skip_rate > 0.5:
            print(f"Warning: High skip rate ({skip_rate:.2%}). Consider reducing learning rate.")
            # Optionally reduce learning rate more aggressively
            if skip_rate > 0.8:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                    print(f"Emergency learning rate reduction to {param_group['lr']}")

# Run training
try:
    # Log parameters to wandb
    wandb.config.update(config)
    
    print(f"Starting training on device: {device}")
    print(f"Source dataset size: {len(source_dataset)}")
    print(f"Target dataset size: {len(target_dataset)}")
    print(f"Validation source dataset size: {len(val_source_dataset)}")
    print(f"Validation target dataset size: {len(val_target_dataset)}")
    
    train(
        model=model,
        source_loader=source_loader,
        target_loader=target_loader,
        val_source_loader=val_source_loader,
        val_target_loader=val_target_loader,
        optimizer=optimizer,
        device=device,
        config=config
    )
    
    print("Training complete!")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config["checkpoint_dir"], "model_final.pth"))
    wandb.save(os.path.join(config["checkpoint_dir"], "model_final.pth"))
    
except Exception as e:
    print(f"Error during training: {e}")
    raise e
finally:
    # Close wandb run
    wandb.finish()
