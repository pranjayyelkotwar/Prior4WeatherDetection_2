# Prior4WeatherDetection: Domain Adaptive Object Detection for Adverse Weather Conditions

## Overview
This repository contains an implementation of a domain adaptive object detection model that leverages prior information for detecting objects in adverse weather conditions, specifically foggy environments. The implementation is based on the approach described in the paper "P4 Adaptive Object Detection" where prior information is used to help bridge the domain gap between clean and foggy images.

## Key Features
- Domain adaptation from clean (source) to foggy (target) images
- Prior-guided adversarial learning for domain adaptation
- Residual feature recovery for enhanced feature extraction
- Integration of dark channel prior for transmission map estimation
- Implementation of Faster R-CNN with domain adaptive components

## Dataset
This implementation uses the Cityscapes dataset with its foggy counterpart (Cityscapes-Foggy):
- **Cityscapes**: Clean urban street scenes
- **Cityscapes-Foggy**: The same scenes with synthetic fog at different density levels (β = 0.01, 0.02, 0.05)

### Dataset Preparation
The dataset can be downloaded using the provided script:
```bash
bash dataset_download.sh
```

This script will:
1. Log in to the Cityscapes website
2. Download the necessary dataset files
3. Extract them to the appropriate directory structure

## Model Architecture
The model is built on Faster R-CNN with the following key components:
- Custom VGG16 backbone
- Prior-aware domain adaptation modules
- Gradient Reversal Layer for adversarial training
- Residual Features Recovery Block

## Installation

### Requirements
See `requirements.txt` for a complete list of dependencies. Install them using:
```bash
pip install -r requirements.txt
```

### Setup
1. Clone this repository
2. Install dependencies
3. Download the dataset using the provided script
4. Run the training loop

## Usage

### Training
To train the model:
```bash
python training_loop_pj.py
```

This will train the model on both source (clean) and target (foggy) domains using the domain adaptation framework.

### Testing
To test the model's forward pass functionality:
```bash
python test_forward_pass.py
```

To visualize the dataset and detection results:
```bash
python test_dataset.py
```

### Visualization
The `visualization_output` directory contains example visualizations of the dataset and detection results.

## Configuration
The training settings are configured within the respective training files. Key hyperparameters include:
- Batch size: 16
- Number of epochs: 25
- Learning rate: 0.0005
- Learning rate decay factor: 0.1
- Gradient clipping norm: 5.0

## Results
The model is evaluated using the mean Average Precision (mAP) metric on both source and target domains.

## Implementation Details
- **Backbone**: Custom VGG16 with frozen weights up to conv3
- **Domain Adaptation**: Gradient reversal layer with prior-guided adversarial loss
- **Classes**: 8 categories (person, rider, car, truck, bus, train, motorcycle, bicycle)
- **Prior Calculation**: Dark channel prior-based transmission map estimation

## Acknowledgements
This implementation is based on the paper "P4 Adaptive Object Detection" and leverages elements from:
- Faster R-CNN
- Dark Channel Prior for defogging
- Domain-Adversarial Neural Networks

## License
[Specify the license information here]

## Citation
If you use this codebase, please cite the original paper:
```
[Citation information for the original paper]
```