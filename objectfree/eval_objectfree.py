import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter, maximum_filter

def compute_local_entropy_gpu(image, window_size=5):
    """
    Compute local entropy for each pixel in the grayscale image using a sliding window on GPU (if available) with PyTorch.
    
    Parameters:
    - image: 2D numpy array (grayscale image)
    - window_size: int, size of the square window (odd number recommended)
    
    Returns:
    - entropy_map: 2D numpy array with entropy values for each pixel
    """
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D array).")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    pad = window_size // 2
    image_padded = np.pad(image, pad, mode='constant')
    image_t = torch.from_numpy(image_padded).float().to(device)
    
    # Unfold to get sliding windows
    patches = image_t.unfold(0, window_size, 1).unfold(1, window_size, 1)
    
    # Slice to match the number of original pixels (handles even/odd window sizes)
    h, w = image.shape
    patches = patches[:h, :w, :, :]
    
    patch_size = window_size ** 2
    num_patches = h * w
    
    # Flatten patches
    patches_flat = patches.reshape(num_patches, patch_size).long()  # Assuming 8-bit grayscale (0-255)
    
    # Compute histograms using scatter_add
    bins = 256
    hist = torch.zeros(num_patches, bins, device=device)
    ones = torch.ones_like(patches_flat, dtype=torch.float, device=device)
    hist.scatter_add_(1, patches_flat, ones)
    
    # Normalize to probabilities
    hist += 1e-10  # Avoid log(0)
    prob = hist / hist.sum(dim=1, keepdim=True)
    
    # Compute entropy
    entropy = - (prob * torch.log2(prob)).sum(dim=1)
    
    # Reshape back to image shape
    entropy_map = entropy.reshape(h, w).cpu().numpy()
    
    return entropy_map

# Example usage:
# Load an image (replace 'path_to_your_image.jpg' with your image file path)
image_color = plt.imread('/home/serverai/ltdoanh/LayoutGeneration/objectfree/14653_keyframes/0002_clip11_frame071_14653_11.jpg')

# Convert to grayscale if necessary
if len(image_color.shape) == 3:
    image_gray = np.mean(image_color, axis=2).astype(np.uint8)
else:
    image_gray = image_color

# Compute entropy map on GPU (falls back to CPU if no GPU)
entropy_map = compute_local_entropy_gpu(image_gray, window_size=20)

# Normalize entropy_map to 0-1
entropy_map_norm = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-10)  # Avoid division by zero

# Smooth the normalized entropy map to make it blurry like the example
sigma = 10  # Adjust sigma for more/less smoothness (higher = more blur)
smoothed_entropy = gaussian_filter(entropy_map_norm, sigma=sigma)

# Detect multiple navigation points: local maxima above a threshold
# Use maximum_filter to find local maxima
footprint = np.ones((3, 3))  # Small neighborhood for local max detection
local_max = smoothed_entropy == maximum_filter(smoothed_entropy, footprint=footprint)

# Threshold to select significant peaks (e.g., above 70% of max)
threshold = 0.7 * smoothed_entropy.max()
significant_peaks = np.where((local_max) & (smoothed_entropy > threshold))

# Get coordinates of peaks (limit to top N if too many)
peaks = list(zip(significant_peaks[1], significant_peaks[0]))  # (x, y)
N_max = 5  # Max number of navigation points to show
peaks = peaks[:N_max]

# Visualize and save results with title and colorbar (instead of showing, since using SSH)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# Plot the smoothed entropy heatmap
im = ax.imshow(smoothed_entropy, cmap='hot')
ax.set_title('Entropy Heatmap: Areas with Concentrated Drawing Strokes')
ax.axis('off')

# Add colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Normalized Entropy (Higher = More Complex/Details)')

# Add multiple navigation points (markers) at significant peaks
for i, (x, y) in enumerate(peaks):
    ax.plot(x, y, 'bo', markersize=10, markeredgecolor='white')  # Blue circle markers
    ax.text(x + 10, y, f'Key Area {i+1}', color='blue', fontsize=10, ha='left')

plt.tight_layout()
plt.savefig('entropy_heatmap_with_multiple_nav_points.png')  # Save the figure
plt.close(fig)  # Close the figure to free memory

# Optionally, save the smoothed entropy as image without extras
plt.imsave('smoothed_entropy_heatmap.png', smoothed_entropy, cmap='hot')

# Optional: Print some stats
mean_entropy = np.mean(smoothed_entropy)
std_entropy = np.std(smoothed_entropy)
print(f"Mean normalized entropy: {mean_entropy:.2f}")
print(f"Std normalized entropy: {std_entropy:.2f}")
print(f"Number of navigation points: {len(peaks)}")
print(f"Navigation points (x, y): {peaks}")
print("Results saved as: entropy_heatmap_with_multiple_nav_points.png, smoothed_entropy_heatmap.png")