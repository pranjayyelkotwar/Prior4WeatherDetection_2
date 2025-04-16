from numpy._typing._array_like import NDArray
from numpy import float64, generic
from typing import Any
import numpy as np
import cv2
from cv2 import ximgproc
from scipy.ndimage import minimum_filter

def estimate_transmission_map(hazy_image, window_size=15, omega=0.95, t0=0.1):
    """
    Estimate transmission map using Dark Channel Prior
    
    Args:
        hazy_image: Input hazy image (0-255 uint8 or 0-1 float)
        window_size: Size of the local patch for dark channel computation
        omega: Parameter to keep slight haze for distant objects (0<omega<=1)
        t0: Lower bound for transmission (avoid division by zero)
    
    Returns:
        transmission: Estimated transmission map (0-1 float32)
    """
    # Convert to float32 in range [0,1] if needed
    if hazy_image.dtype == np.uint8:
        hazy_image = hazy_image.astype(np.float32) / 255.0
    
    # Step 1: Estimate atmospheric light (A)
    # Compute dark channel
    dark_channel: NDArray[float64] | NDArray[Any] | NDArray[generic] | Any = get_dark_channel(hazy_image, window_size)
    
    # Select top 0.1% brightest pixels in dark channel
    num_pixels = int(dark_channel.size * 0.001)
    indices = np.argpartition(dark_channel.ravel(), -num_pixels)[-num_pixels:]
    
    # Find corresponding pixels in original image
    hazy_pixels = hazy_image.reshape(-1, 3)[indices]
    atmospheric_light = np.max(hazy_pixels, axis=0)
    
    # Step 2: Normalize the hazy image by atmospheric light
    normalized_hazy = np.zeros_like(hazy_image)
    for c in range(3):
        normalized_hazy[:,:,c] = hazy_image[:,:,c] / atmospheric_light[c]
    
    # Step 3: Compute dark channel of normalized image
    dark_channel_norm = get_dark_channel(normalized_hazy, window_size)
    
    # Step 4: Estimate transmission
    transmission = 1.0 - omega * dark_channel_norm.astype(np.float32)
    
    # Apply lower bound to transmission
    transmission = np.clip(transmission, t0, 1.0)
    
    return transmission, atmospheric_light

def get_dark_channel(image, window_size=15):
    """
    Compute dark channel prior for an image
    
    Args:
        image: Input image (H,W,3)
        window_size: Size of the local patch
    
    Returns:
        dark_channel: Dark channel image (H,W)
    """
    # Take minimum over color channels
    min_channel = np.min(image, axis=2)
    
    # Apply minimum filter (local minimum operation)
    dark_channel = minimum_filter(min_channel, size=window_size, 
                                mode='reflect')
    
    return dark_channel

def refine_transmission(transmission, hazy_image, guided_filter_radius=60, eps=1e-3):
    # Convert hazy image to grayscale to use as guide
    if hazy_image.dtype == np.float32 and np.max(hazy_image) <= 1.0:
        gray = cv2.cvtColor((hazy_image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        gray = cv2.cvtColor(hazy_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
    # Apply guided filter
    if hasattr(cv2, 'ximgproc'):
        refined_transmission = cv2.ximgproc.guidedFilter(
            guide=gray,
            src=transmission,
            radius=guided_filter_radius,
            eps=eps
        )
    else:
        # Fallback to a less efficient but common implementation
        radius = int(guided_filter_radius)
        transmission_blurred = cv2.blur(transmission, (radius, radius))
        refined_transmission = transmission_blurred
    
    return refined_transmission