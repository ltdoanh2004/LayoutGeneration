import cv2
import numpy as np
import os
import glob

def create_image_grid(image_paths, grid_size, output_path):
    """
    Create a grid of images and save it to the specified output path.

    :param image_paths: List of paths to the images.
    :param grid_size: Tuple (rows, cols) specifying the grid size.
    :param output_path: Path to save the output grid image.
    """
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image at {img_path}")
        else:
            images.append(img)
    
    if not images:
        raise ValueError("No valid images found to create a grid.")

    # Check if all images have the same size
    img_height, img_width, _ = images[0].shape
    for img in images:
        if img.shape != (img_height, img_width, 3):
            raise ValueError("All images must have the same dimensions")

    rows, cols = grid_size
    total_cells = rows * cols

    # Add black images if not enough images to fill the grid
    while len(images) < total_cells:
        images.append(np.zeros((img_height, img_width, 3), dtype=np.uint8))

    # Create an empty array to hold the grid
    grid_image = np.zeros((img_height * rows, img_width * cols, 3), dtype=np.uint8)

    # Fill the grid with images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width, :] = img

    # Save the grid image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, grid_image)
    print(f"Grid image saved to {output_path}")

# Example usage
image_folder = r'/home/serverai/ltdoanh/LayoutGeneration/outputs/Dang/pipeline/pipeline/lpips/run_tv2_lpips_v11/keyframes'
image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))  # Get all .jpg files in the folder

# Ensure the number of images matches the grid size
num_images = len(image_paths)
grid_size = (6, 6)  # 6x6 grid

# Select only the required number of images
image_paths = image_paths[:grid_size[0] * grid_size[1]]
print("Selected image paths:", image_paths)

output_path = 'outputs/visualize/v11_keyframes.jpg'
create_image_grid(image_paths, grid_size, output_path)
