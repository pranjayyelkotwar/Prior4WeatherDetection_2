from math import isinf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN 
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from models.prior_estimation_network import PriorEstimationNetwork
from models.residual_features_recovery_block import ResidualFeatureRecoveryBlock
from models.gradient_reversal_layer import GradientReversal
from models.custom_vgg_backbone import CustomVGGBackbone



class DomainAdaptiveFasterRCNN(nn.Module):
    def __init__(self, num_classes, backbone_name='vgg16' , verbose=False):
        super().__init__()
        self.verbose = verbose  
        # Base detection network
        if backbone_name == 'resnet50':
            self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        else:  # Default to VGG16
            # self.backbone = self._build_vgg_backbone()
            self.backbone = CustomVGGBackbone(verbose=self.verbose)

        # Configure anchor generator for RPN - this is a key fix
        anchor_generator = AnchorGenerator(
                            sizes=((32,), (64,)),
                            aspect_ratios=((0.5, 1.0, 2.0),) * 2
                        )


        
        # Configure the RoI pooling - this is a key fix
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0','1'],  # Match the feature map keys from your backbone
            output_size=7,
            sampling_ratio=2
        )
        
        # Create the detector with the custom configurations
        self.detector = FasterRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
        
        # Prior Estimation Networks (PEN)
        self.pen_c4 = PriorEstimationNetwork(in_channels=512 , verbose=self.verbose)  # For conv4 features
        self.pen_c5 = PriorEstimationNetwork(in_channels=512 , verbose=self.verbose)  # For conv5 features
        
        # Residual Feature Recovery Blocks (RFRB)
        self.rfrb_c4 = ResidualFeatureRecoveryBlock(in_channels=256, out_channels=512 , verbose=self.verbose)
        self.rfrb_c5 = ResidualFeatureRecoveryBlock(in_channels=512, out_channels=512, verbose=self.verbose)
        
        # Gradient reversal layer for adversarial training
        self.grl = GradientReversal(lambda_=0.1)  # Reduce lambda for more stable training
        
        # Loss weights
        self.pal_weight = 2  # Weight for prior adversarial loss
        self.reg_weight = 1  # Weight for regularization loss
        
    def forward(self, images, prior_images, targets):
        if self.verbose:
            print(f'shape of images is {images.shape}')
            print(f'shape of prior images is {prior_images.shape}')
        # Feature extraction
        if self.verbose:
            print('passing images through vgg16 backbone')
        features = self.backbone(images)
        if self.verbose:
            print('finished passing images through vgg16 backbone')

        if self.training:
            # Training mode - compute all losses
            losses = {}

            # Get features from intermediate layers
            c3_features = self.backbone.get_c3_features()
            c4_features = features['0'] 
            c5_features = features['1'] 

            # Apply RFRBs for target domain
            if not targets[0]['is_source']:
                if self.verbose:
                    print('passing c4_features through rfrb_c4 for target domain')
                rfrb_c4_features = self.rfrb_c4(c3_features)
                # Ensure rfrb_c4_features has the same spatial dimensions as c4_features
                if rfrb_c4_features.shape[-2:] != c4_features.shape[-2:]:
                    rfrb_c4_features = F.interpolate(
                        rfrb_c4_features, 
                        size=(c4_features.shape[-2], c4_features.shape[-1]), 
                        mode='bilinear', 
                        align_corners=False
                    )
                if self.verbose:
                    print(f'shape of rfrb_c4_features is {rfrb_c4_features.shape}')
                    print(f'shape of c4_features is {c4_features.shape}')
                # Fix: Replace in-place addition with out-of-place operation
                c4_features = c4_features + rfrb_c4_features
                features['0'] = c4_features  # Use recovered features
                rfrb_c5_features = self.rfrb_c5(c4_features)
                # Ensure rfrb_c5_features has the same spatial dimensions as c5_features
                if rfrb_c5_features.shape[-2:] != c5_features.shape[-2:]:
                    rfrb_c5_features = F.interpolate(
                        rfrb_c5_features, 
                        size=(c5_features.shape[-2], c5_features.shape[-1]), 
                        mode='bilinear', 
                        align_corners=False
                    )
                # Fix: Replace in-place addition with out-of-place operation
                c5_features = c5_features + rfrb_c5_features
                features['1'] = c5_features  # Use recovered features
                if self.verbose:
                    print('finished passing c4_features through rfrb_c4')

            # Detection losses (for source domain)
            if targets[0]['is_source']:
                if self.verbose:
                    print('passing images through detector for source domain')
                detector_targets = [
                    {k: v for k, v in t.items() if k in ['boxes', 'labels', 'image_id', 'area', 'iscrowd']}
                    for t in targets if t['is_source']
                ]
                try:
                    det_losses = self.detector(images, detector_targets)
                    if self.verbose:
                        print(f'det loss : {det_losses}')
                        print('finished passing images through detector for source domain')
                    # Scale detection losses to prevent explosion
                    for k, v in det_losses.items():
                        # Apply gradient clipping by value for detection losses
                        if torch.isnan(v):
                            det_losses[k] = torch.tensor(0.001, device=v.device, requires_grad=True)
                        elif torch.isinf(v):
                            # Clip excessively large losses
                            det_losses[k] = torch.clamp(v, max=100.0)
                    
                    losses.update(det_losses)
                except Exception as e:
                    if self.verbose:
                        print(f"Error in detection loss: {e}")
                
                if self.verbose:
                    print('finished passing images through detector for source domain')

            # Prior adversarial losses (for both domains)
            try:
                pal_loss = 0.0
                if self.verbose:
                    print('passing c4_features through pen_c4 for prior prediction')
                
                # Apply gradient reversal layer
                c4_features_grl = self.grl(c4_features)
                c5_features_grl = self.grl(c5_features)
                
                # Get prior predictions
                pred_prior_c4 = self.pen_c4(c4_features_grl)
                pred_prior_c5 = self.pen_c5(c5_features_grl)
                
                if self.verbose:
                    print('finished passing c4_features through pen_c4 for prior prediction')

                # Resize prior images to match feature map dimensions
                c4_size = (pred_prior_c4.size(2), pred_prior_c4.size(3))  # (U, V) for conv4
                c5_size = (pred_prior_c5.size(2), pred_prior_c5.size(3))  # (U, V) for conv5

                # Resize priors to match c4 and c5 feature map sizes
                prior_image_c4 = F.interpolate(prior_images, size=c4_size, mode='bilinear', align_corners=False)
                prior_image_c5 = F.interpolate(prior_images, size=c5_size, mode='bilinear', align_corners=False)

                # Normalize priors to [-1, 1] to match Tanh output of PEN
                # Use robust normalization with epsilon to prevent division by zero
                epsilon = 1e-5
                prior_image_c4 = 2.0 * ((prior_image_c4 - prior_image_c4.min()) / 
                                     (prior_image_c4.max() - prior_image_c4.min() + epsilon)) - 1.0
                prior_image_c5 = 2.0 * ((prior_image_c5 - prior_image_c5.min()) / 
                                     (prior_image_c5.max() - prior_image_c5.min() + epsilon)) - 1.0
                
                # Clamp to ensure values stay in [-1, 1]
                prior_image_c4 = torch.clamp(prior_image_c4, -1.0, 1.0)
                prior_image_c5 = torch.clamp(prior_image_c5, -1.0, 1.0)

                if self.verbose:
                    print(f'shape of pred prior c4 is {pred_prior_c4.shape}')
                    print(f'shape of prior image c4 is {prior_image_c4.shape}')

                # Ensure prior_image_* has 3 channels to match pred_prior_*
                if prior_image_c4.size(1) == 1:
                    prior_image_c4 = prior_image_c4.repeat(1, 3, 1, 1)
                    prior_image_c5 = prior_image_c5.repeat(1, 3, 1, 1)
                
                # Use smooth L1 loss instead of MSE for robustness
                pal_loss += F.smooth_l1_loss(pred_prior_c4, prior_image_c4)
                pal_loss += F.smooth_l1_loss(pred_prior_c5, prior_image_c5)
                
                # Weight the PAL loss
                losses['loss_pal'] = self.pal_weight * pal_loss
                
            except Exception as e:
                if self.verbose:
                    print(f"Error in PAL loss: {e}")
                losses['loss_pal'] = torch.tensor(0.0, device=c4_features.device, requires_grad=True)

            # Regularization loss for target domain
            if not targets[0]['is_source']:
                try:
                    
                    c4_reg = torch.mean(torch.abs(rfrb_c4_features))
                    c5_reg = torch.mean(torch.abs(rfrb_c5_features))
                    reg_loss = c4_reg + c5_reg
                    
                    # Apply weight to regularization loss
                    losses['loss_reg'] = self.reg_weight * reg_loss
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error in regularization loss: {e}")
                    losses['loss_reg'] = torch.tensor(0.0, device=c4_features.device, requires_grad=True)

            return losses
        else:
            return self.detector(images)
