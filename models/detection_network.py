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

# Enable cudnn benchmarking and deterministic algorithms for performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False  # Set to False for maximum speed

class DomainAdaptiveFasterRCNN(nn.Module):
    def __init__(self, num_classes, backbone_name='vgg16', verbose=False):
        super().__init__()
        self.verbose = verbose  
        # Base detection network
        if backbone_name == 'resnet50':
            self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        else:  # Default to VGG16
            # self.backbone = self._build_vgg_backbone()
            self.backbone = CustomVGGBackbone(verbose=self.verbose)

        # Configure anchor generator for RPN with fewer anchors for speed
        anchor_generator = AnchorGenerator(
                            sizes=((32,), (64,)),
                            aspect_ratios=((0.5, 1.0, 2.0),) * 2
                        )
        
        # Configure the RoI pooling with simplified parameters
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0','1'],  # Match the feature map keys from your backbone
            output_size=7,
            sampling_ratio=0  # Use adaptive sampling (faster)
        )
        
        # Create the detector with the custom configurations
        self.detector = FasterRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            rpn_pre_nms_top_n_train=1000,  # Reduce number of proposals for speed (was 2000)
            rpn_pre_nms_top_n_test=500,    # Reduce number of proposals for speed (was 1000)
            rpn_post_nms_top_n_train=500,  # Reduce number of proposals for speed (was 1000)
            rpn_post_nms_top_n_test=100,   # Reduce number of proposals for speed (was 300)
        )
        
        # Prior Estimation Networks (PEN) - simplify to improve speed
        self.pen_c4 = PriorEstimationNetwork(in_channels=512, verbose=self.verbose)  
        self.pen_c5 = PriorEstimationNetwork(in_channels=512, verbose=self.verbose)  
        
        # Residual Feature Recovery Blocks (RFRB)
        self.rfrb_c4 = ResidualFeatureRecoveryBlock(in_channels=256, out_channels=512, verbose=self.verbose)
        self.rfrb_c5 = ResidualFeatureRecoveryBlock(in_channels=512, out_channels=512, verbose=self.verbose)
        
        # Gradient reversal layer for adversarial training
        self.grl = GradientReversal(lambda_=0.1)  # Reduce lambda for more stable training
        
        # Loss weights
        self.pal_weight = 2  # Weight for prior adversarial loss
        self.reg_weight = 1  # Weight for regularization loss
        
        # Cache for feature maps to avoid recomputation
        self.feature_cache = {}
        
    def forward(self, images, prior_images, targets):
        if self.verbose:
            print(f'shape of images is {images.shape}')
            if prior_images is not None:
                print(f'shape of prior images is {prior_images.shape}')
                
        # Clear feature cache at the beginning of forward pass
        self.feature_cache = {}
        
        # Feature extraction
        if self.verbose:
            print('passing images through vgg16 backbone')
        features = self.backbone(images)
        if self.verbose:
            print('finished passing images through vgg16 backbone')

        if self.training:
            # Training mode - compute all losses
            losses = {}

            # Get features from intermediate layers (with caching)
            c3_features = self.backbone.get_c3_features()
            c4_features = features['0'] 
            c5_features = features['1']
            
            # Cache these features
            self.feature_cache['c3'] = c3_features
            self.feature_cache['c4'] = c4_features
            self.feature_cache['c5'] = c5_features

            # Apply RFRBs for target domain
            if targets is not None and not targets[0]['is_source']:
                if self.verbose:
                    print('passing c4_features through rfrb_c4 for target domain')
                
                # Apply RFRB to C4 features
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
                
                # Cache rfrb_c4_features for regularization loss
                self.feature_cache['rfrb_c4'] = rfrb_c4_features
                
                # Apply RFRB to C5 features
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
                
                # Cache rfrb_c5_features for regularization loss
                self.feature_cache['rfrb_c5'] = rfrb_c5_features
                
                if self.verbose:
                    print('finished passing c4_features through rfrb_c4')

            # Detection losses (for source domain)
            if targets is not None and targets[0]['is_source']:
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
                        
                    # Handle invalid detection losses
                    for k, v in det_losses.items():
                        if torch.isnan(v):
                            det_losses[k] = torch.tensor(0.001, device=v.device, requires_grad=True)
                        elif torch.isinf(v):
                            # Clip excessively large losses
                            det_losses[k] = torch.clamp(v, max=100.0)
                    
                    losses.update(det_losses)
                except Exception as e:
                    if self.verbose:
                        print(f"Error in detection loss: {e}")

            # Prior adversarial losses (for both domains)
            if prior_images is not None:
                try:
                    pal_loss = 0.0
                    if self.verbose:
                        print('passing c4_features through pen_c4 for prior prediction')
                    
                    # Apply gradient reversal layer
                    c4_features_grl = self.grl(c4_features)
                    c5_features_grl = self.grl(c5_features)
                    
                    # Get prior predictions (use caching)
                    if 'pred_prior_c4' not in self.feature_cache:
                        self.feature_cache['pred_prior_c4'] = self.pen_c4(c4_features_grl)
                    if 'pred_prior_c5' not in self.feature_cache:
                        self.feature_cache['pred_prior_c5'] = self.pen_c5(c5_features_grl)
                    
                    pred_prior_c4 = self.feature_cache['pred_prior_c4']
                    pred_prior_c5 = self.feature_cache['pred_prior_c5']
                    
                    if self.verbose:
                        print('finished passing c4_features through pen_c4 for prior prediction')

                    # Size for feature maps
                    c4_size = (pred_prior_c4.size(2), pred_prior_c4.size(3))
                    c5_size = (pred_prior_c5.size(2), pred_prior_c5.size(3))

                    # Interpolate prior images (once) and cache them
                    if 'prior_image_c4' not in self.feature_cache:
                        # Resize priors to match feature map sizes
                        prior_image_c4 = F.interpolate(prior_images, size=c4_size, mode='bilinear', align_corners=False)
                        prior_image_c5 = F.interpolate(prior_images, size=c5_size, mode='bilinear', align_corners=False)
                        
                        # Normalize priors to [-1, 1] to match Tanh output of PEN
                        epsilon = 1e-5
                        prior_image_c4 = 2.0 * ((prior_image_c4 - prior_image_c4.min()) / 
                                             (prior_image_c4.max() - prior_image_c4.min() + epsilon)) - 1.0
                        prior_image_c5 = 2.0 * ((prior_image_c5 - prior_image_c5.min()) / 
                                             (prior_image_c5.max() - prior_image_c5.min() + epsilon)) - 1.0
                        
                        # Clamp to ensure values stay in [-1, 1]
                        prior_image_c4 = torch.clamp(prior_image_c4, -1.0, 1.0)
                        prior_image_c5 = torch.clamp(prior_image_c5, -1.0, 1.0)
                        
                        # Ensure prior_image_* has 3 channels to match pred_prior_*
                        if prior_image_c4.size(1) == 1:
                            prior_image_c4 = prior_image_c4.repeat(1, 3, 1, 1)
                            prior_image_c5 = prior_image_c5.repeat(1, 3, 1, 1)
                            
                        # Cache the processed prior images
                        self.feature_cache['prior_image_c4'] = prior_image_c4
                        self.feature_cache['prior_image_c5'] = prior_image_c5
                    
                    # Use cached prior images
                    prior_image_c4 = self.feature_cache['prior_image_c4']
                    prior_image_c5 = self.feature_cache['prior_image_c5']

                    if self.verbose:
                        print(f'shape of pred prior c4 is {pred_prior_c4.shape}')
                        print(f'shape of prior image c4 is {prior_image_c4.shape}')
                    
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
            if targets is not None and not targets[0]['is_source']:
                try:
                    # Use cached RFRB features
                    rfrb_c4_features = self.feature_cache['rfrb_c4']
                    rfrb_c5_features = self.feature_cache['rfrb_c5']
                    
                    c4_reg = torch.mean(torch.abs(rfrb_c4_features))
                    c5_reg = torch.mean(torch.abs(rfrb_c5_features))
                    reg_loss = c4_reg + c5_reg
                    
                    # Apply weight to regularization loss
                    losses['loss_reg'] = self.reg_weight * reg_loss
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error in regularization loss: {e}")
                    losses['loss_reg'] = torch.tensor(0.0, device=c4_features.device, requires_grad=True)

            # Clean up cached tensors that won't be needed anymore
            self.feature_cache.clear()
            return losses
        else:
            # Inference mode - just run detector
            return self.detector(images)
