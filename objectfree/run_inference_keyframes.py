import os
import glob
import torch
from inference_dino import LoadDetector

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Đường dẫn folder chứa keyframes
    keyframes_folder = "/home/serverai/ltdoanh/LayoutGeneration/14653_keyframes"
    
    # Lấy tất cả ảnh .jpg trong folder (trừ preview)
    image_paths = sorted(glob.glob(os.path.join(keyframes_folder, "*.jpg")))
    # Loại bỏ preview image
    image_paths = [p for p in image_paths if "preview" not in os.path.basename(p).lower()]
    
    if len(image_paths) == 0:
        print(f"No images found in {keyframes_folder}")
        return
    
    print(f"Found {len(image_paths)} keyframes to process:")
    for i, p in enumerate(image_paths[:5]):  # show first 5
        print(f"  {i}: {os.path.basename(p)}")
    if len(image_paths) > 5:
        print(f"  ... and {len(image_paths) - 5} more")
    
    # Tạo output directory
    output_dir = "/home/serverai/ltdoanh/LayoutGeneration/outputs_14653"
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo detector
    detector = LoadDetector(
        config_path="config.yaml",
        checkpoint_path="./Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt",
        image_path=image_paths,
        device=device,
        batch_size=len(image_paths),  # xử lý tất cả ảnh
        output_dir=output_dir
    )
    
    print("\n" + "="*60)
    print("Starting inference...")
    print("="*60 + "\n")
    
    results = detector.forward()
    
    print("\n" + "="*60)
    print("Detection completed!")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    print(f"Results saved in: {output_dir}/")
    print(f"JSON results: {output_dir}/detection_results.json")
    print("\nAnnotated images:")
    for i in range(min(3, len(results))):
        if results[i].get("annotated_image"):
            print(f"  - {results[i]['annotated_image']}")
    if len(results) > 3:
        print(f"  ... and {len(results) - 3} more")

if __name__ == "__main__":
    main()
