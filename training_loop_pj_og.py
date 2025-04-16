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

from models.detection_network import DomainAdaptiveFasterRCNN
from utils.data.cityscapes_clean_dataset import Cityscapes_Clean_Dataset, cityscapes_clean_dataset_collate_fn
from utils.data.cityscapes_foggy_dataset import Cityscapes_Foggy_Dataset, cityscapes_foggy_dataset_collate_fn
from utils.evaluation import evaluate_detection
import gc

batch_size = 16
num_epochs = 5
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

# Initialize model
model = DomainAdaptiveFasterRCNN(num_classes=8, backbone_name='vgg16', verbose=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Log model architecture to wandb
wandb.watch(model, log="all", log_freq=10)

# Optimizer with a lower learning rate for stability
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)

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

# Dataloaders with smaller batch size for stability
source_loader = DataLoader(
    source_dataset,
    batch_size = batch_size,
    shuffle=True,
    collate_fn=cityscapes_clean_dataset_collate_fn,
    num_workers=4
)

target_loader = DataLoader(
    target_dataset,
    batch_size = batch_size,
    shuffle=True,
    collate_fn=cityscapes_foggy_dataset_collate_fn,
    num_workers=4
)

val_source_loader = DataLoader(
    val_source_dataset,
    batch_size = batch_size,
    shuffle=False,
    collate_fn=cityscapes_clean_dataset_collate_fn,
    num_workers=4
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
    "eval_interval": 1,
    "checkpoint_dir": "./checkpoints",
    "max_grad_norm": 1.0,     # Maximum gradient norm for clipping
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
    # skipped_batches = 0
    # total_batches = 0
    
    # for epoch in range(config["num_epochs"]):
    #     # Training phase
    #     model.train()
    #     source_epoch_loss = 0.0
    #     target_epoch_loss = 0.0
    #     classifier_loss = 0.0
    #     box_reg_loss = 0.0
    #     objectness_loss = 0.0
    #     rpn_box_reg_loss = 0.0
    #     pal_loss = 0.0
    #     reg_loss = 0.0
        
    #     num_batches = 0
    #     epoch_skipped_batches = 0
    #     start_time = time.time()
        
    #     # Create iterator for target_loader to handle different dataset sizes
    #     target_iter = iter(target_loader)
        
    #     source_tqdm = tqdm(source_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train Source]")
        
    #     for batch_idx, source_batch in enumerate(source_tqdm):
    #         total_batches += 1
            
    #         try:
    #             # Get a batch from target dataset
    #             try:
    #                 target_batch = next(target_iter)
    #             except StopIteration:
    #                 target_iter = iter(target_loader)
    #                 target_batch = next(target_iter)
                
    #             # Source domain training
    #             source_images, source_prior_images, source_targets = source_batch
    #             source_images = [img.to(device) for img in source_images]
    #             source_prior_images = [img.to(device) for img in source_prior_images]
    #             source_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in source_targets]

    #             source_images_tensor = torch.stack(source_images)
    #             source_prior_images_tensor = torch.stack(source_prior_images)

    #             # Forward pass for source domain
    #             source_losses = model(source_images_tensor, source_prior_images_tensor, source_targets)
                
    #             # Check for valid losses and replace NaN/Inf values
    #             for k in list(source_losses.keys()):
    #                 if not is_valid_loss(source_losses[k]):
    #                     print(f"Warning: Invalid {k} in source batch {batch_idx}: {source_losses[k]}")
    #                     source_losses[k] = torch.tensor(0.0, device=device, requires_grad=True)
                
    #             # Sum all source losses with upper bound to prevent explosion
    #             source_loss_values = [v.item() if is_valid_loss(v) else 0.0 for v in source_losses.values()]
    #             if any(l > 100.0 for l in source_loss_values):
    #                 print(f"Warning: Large loss values in source batch {batch_idx}: {source_loss_values}")
                
    #             source_loss = sum(source_losses.values())
                
    #             if not is_valid_loss(source_loss):
    #                 raise ValueError(f"Invalid combined source loss: {source_loss}")
                
    #             # Track individual losses (safely)
    #             classifier_loss += source_losses.get('loss_classifier', torch.tensor(0.0)).item()
    #             box_reg_loss += source_losses.get('loss_box_reg', torch.tensor(0.0)).item()
    #             objectness_loss += source_losses.get('loss_objectness', torch.tensor(0.0)).item()
    #             rpn_box_reg_loss += source_losses.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
    #             pal_loss += source_losses.get('loss_pal', torch.tensor(0.0)).item()

    #             # Backward pass and optimization with gradient clipping
    #             optimizer.zero_grad()
    #             source_loss.backward()
    #             # Apply gradient clipping to prevent exploding gradients
    #             utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    #             optimizer.step()
                
    #             source_epoch_loss += source_loss.item()
                
    #             # Target domain training
    #             target_images, target_prior_images, target_targets = target_batch
    #             target_images = [img.to(device) for img in target_images]
    #             target_prior_images = [img.to(device) for img in target_prior_images]
    #             target_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in target_targets]

    #             target_images_tensor = torch.stack(target_images)
    #             target_prior_images_tensor = torch.stack(target_prior_images)

    #             # Forward pass for target domain
    #             target_losses = model(target_images_tensor, target_prior_images_tensor, target_targets)
                
    #             # Check for valid losses and replace NaN/Inf values
    #             for k in list(target_losses.keys()):
    #                 if not is_valid_loss(target_losses[k]):
    #                     print(f"Warning: Invalid {k} in target batch {batch_idx}: {target_losses[k]}")
    #                     target_losses[k] = torch.tensor(0.0, device=device, requires_grad=True)
                
    #             # Sum all target losses with validation
    #             target_loss = sum(target_losses.values())
                
    #             if not is_valid_loss(target_loss):
    #                 raise ValueError(f"Invalid combined target loss: {target_loss}")
                
    #             # Track individual target losses (safely)
    #             pal_loss += target_losses.get('loss_pal', torch.tensor(0.0)).item()
    #             reg_loss += target_losses.get('loss_reg', torch.tensor(0.0)).item()

    #             # Backward pass and optimization with gradient clipping
    #             optimizer.zero_grad()
    #             target_loss.backward()
    #             utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
    #             optimizer.step()
                
    #             target_epoch_loss += target_loss.item()
    #             num_batches += 1
                
    #             # Log per-batch losses to wandb for more granular visualization
    #             global_step = epoch * len(source_loader) + batch_idx
    #             wandb.log({
    #                 "batch/source_loss": source_loss.item(),
    #                 "batch/target_loss": target_loss.item(),
    #                 "batch/classifier_loss": source_losses.get('loss_classifier', torch.tensor(0.0)).item(),
    #                 "batch/box_reg_loss": source_losses.get('loss_box_reg', torch.tensor(0.0)).item(),
    #                 "batch/objectness_loss": source_losses.get('loss_objectness', torch.tensor(0.0)).item(),
    #                 "batch/rpn_box_reg_loss": source_losses.get('loss_rpn_box_reg', torch.tensor(0.0)).item(),
    #                 "batch/pal_loss": (source_losses.get('loss_pal', torch.tensor(0.0)).item() + 
    #                                  target_losses.get('loss_pal', torch.tensor(0.0)).item()) / 2,
    #                 "batch/reg_loss": target_losses.get('loss_reg', torch.tensor(0.0)).item(),
    #                 "global_step": global_step
    #             })
                
    #             # Update tqdm description with latest losses
    #             source_tqdm.set_postfix({
    #                 'source_loss': f"{source_loss.item():.4f}", 
    #                 'target_loss': f"{target_loss.item():.4f}",
    #                 'skipped': epoch_skipped_batches
    #             })
                
    #         except Exception as e:
    #             epoch_skipped_batches += 1
    #             skipped_batches += 1
    #             print(f"Error in batch {batch_idx}: {str(e)}")
    #             print(f"Skipping this batch and continuing training...")
    #             continue
        
    #     # Calculate average losses (safely)
    #     if num_batches > 0:
    #         avg_source_loss = source_epoch_loss / num_batches
    #         avg_target_loss = target_epoch_loss / num_batches
    #         avg_classifier_loss = classifier_loss / num_batches
    #         avg_box_reg_loss = box_reg_loss / num_batches
    #         avg_objectness_loss = objectness_loss / num_batches
    #         avg_rpn_box_reg_loss = rpn_box_reg_loss / num_batches
    #         avg_pal_loss = pal_loss / (num_batches * 2)  # Both source and target contribute
    #         avg_reg_loss = reg_loss / num_batches
    #     else:
    #         print("Warning: No valid batches in epoch. Using zeros for metrics.")
    #         avg_source_loss = avg_target_loss = avg_classifier_loss = avg_box_reg_loss = 0.0
    #         avg_objectness_loss = avg_rpn_box_reg_loss = avg_pal_loss = avg_reg_loss = 0.0
        
    #     # Clear GPU cache to free up memory
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    #         # Optional: print memory stats for debugging
    #         allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    #         reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    #         print(f"  GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

    #     # Perform garbage collection to help free memory
    #     gc.collect()

        
    #     epoch_time = time.time() - start_time
    #     skip_rate = epoch_skipped_batches / (num_batches + epoch_skipped_batches) if (num_batches + epoch_skipped_batches) > 0 else 0
        
    #     print(f"Epoch [{epoch+1}/{config['num_epochs']}] Time: {epoch_time:.2f}s, Skip rate: {skip_rate:.2%}")
    #     print(f"  Source Loss: {avg_source_loss:.4f}, Target Loss: {avg_target_loss:.4f}")
        
    #     # Log to wandb (safely ensuring all values are finite)
    #     wandb.log({
    #         "epoch": epoch + 1,
    #         "train/source_loss": avg_source_loss if is_valid_loss(avg_source_loss) else 0.0,
    #         "train/target_loss": avg_target_loss if is_valid_loss(avg_target_loss) else 0.0,
    #         "train/classifier_loss": avg_classifier_loss if is_valid_loss(avg_classifier_loss) else 0.0,
    #         "train/box_reg_loss": avg_box_reg_loss if is_valid_loss(avg_box_reg_loss) else 0.0,
    #         "train/objectness_loss": avg_objectness_loss if is_valid_loss(avg_objectness_loss) else 0.0,
    #         "train/rpn_box_reg_loss": avg_rpn_box_reg_loss if is_valid_loss(avg_rpn_box_reg_loss) else 0.0,
    #         "train/pal_loss": avg_pal_loss if is_valid_loss(avg_pal_loss) else 0.0,
    #         "train/reg_loss": avg_reg_loss if is_valid_loss(avg_reg_loss) else 0.0,
    #         "train/epoch_time": epoch_time,
    #         "train/skip_rate": skip_rate,
    #         "lr": optimizer.param_groups[0]['lr']
    #     })
        
        # # Validation phase
        # if (epoch + 1) % config["eval_interval"] == 0:
        #     print("Running validation...")
        #     try:
        #         # Evaluate on source validation set
        #         source_metrics = evaluate_detection(
        #             model, val_source_loader, device, num_classes=8
        #         )
        #         # Evaluate on target validation set
        #         target_metrics = evaluate_detection(
        #             model, val_target_loader, device, num_classes=8
        #         )

        #         print('calculated target metrics')
        #         source_mAP = source_metrics['mAP']
        #         target_mAP = target_metrics['mAP']
                
        #         print(f"  Validation: Source mAP: {source_mAP:.4f}, Target mAP: {target_mAP:.4f}")
                
        #         # Log per-class AP to wandb
        #         source_ap_per_class = source_metrics['AP_per_class']
        #         target_ap_per_class = target_metrics['AP_per_class']
                
        #         for class_id, ap in source_ap_per_class.items():
        #             wandb.log({f"val_source/AP_class_{class_id}": ap, "epoch": epoch + 1})
                
        #         for class_id, ap in target_ap_per_class.items():
        #             wandb.log({f"val_target/AP_class_{class_id}": ap, "epoch": epoch + 1})
                
        #         # Log overall metrics
        #         wandb.log({
        #             "val_source/mAP": source_mAP,
        #             "val_target/mAP": target_mAP,
        #             "epoch": epoch + 1
        #         })
        #     except Exception as e:
        #         print(f"Error during validation: {e}")
        #         print("Skipping this validation step and continuing training...")

        
        # # Save checkpoint
        # if (epoch + 1) % config["save_interval"] == 0:
        #     try:
        #         save_checkpoint(epoch + 1, model, optimizer, avg_source_loss + avg_target_loss)
        #         # Also save model to wandb
        #         wandb.save(os.path.join(config["checkpoint_dir"], f"model_epoch_{epoch+1}.pth"))
        #     except Exception as e:
        #         print(f"Error saving checkpoint: {e}")
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