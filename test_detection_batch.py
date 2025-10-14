#!/usr/bin/env python3
"""
Test object detection on multiple keyframe folders
"""
import os
import sys
import torch
from objectfree.inference_dino import LoadDetector

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Base folder containing all keyframe folders
    base_folder = "/home/serverai/ltdoanh/LayoutGeneration/samples/keyframe4check"
    
    # Get all keyframe folders (ending with _keyframes)
    keyframe_folders = []
    for item in sorted(os.listdir(base_folder)):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path) and item.endswith("_keyframes"):
            keyframe_folders.append(item_path)
    
    if len(keyframe_folders) == 0:
        print(f"No keyframe folders found in {base_folder}")
        return
    
    print(f"\nFound {len(keyframe_folders)} keyframe folders:")
    for i, folder in enumerate(keyframe_folders):
        print(f"  {i+1}. {os.path.basename(folder)}")
    
    # Process each folder
    for idx, keyframes_folder in enumerate(keyframe_folders):
        folder_name = os.path.basename(keyframes_folder)
        print(f"\n{'='*70}")
        print(f"Processing folder {idx+1}/{len(keyframe_folders)}: {folder_name}")
        print(f"{'='*70}")
        
        # Create output directory for this folder
        output_dir = f"/home/serverai/ltdoanh/LayoutGeneration/outputs_detection/{folder_name}_detections"
        
        try:
            # Initialize detector
            detector = LoadDetector(
                config_path="objectfree/config.yaml",
                checkpoint_path="/home/serverai/ltdoanh/LayoutGeneration/objectfree/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt",
                image_path=None,  # Will be set in run_inference
                device=device,
                batch_size=1,
                output_dir=output_dir
            )
            
            # Run inference
            results = detector.run_inference(
                keyframes_folder=keyframes_folder,
                output_dir=output_dir
            )
            
            print(f"\n✅ Successfully processed {folder_name}")
            print(f"   Results saved to: {output_dir}")
            
        except Exception as e:
            print(f"\n❌ Error processing {folder_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print("All folders processed!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
