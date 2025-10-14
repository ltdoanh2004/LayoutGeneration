#!/usr/bin/env python3
"""
Test object detection on a single keyframe folder
"""
import os
import sys
import torch
from objectfree.inference_dino import LoadDetector

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test with one folder first
    keyframes_folder = "/home/serverai/ltdoanh/LayoutGeneration/samples/keyframe4check/14653_keyframes"
    output_dir = "/home/serverai/ltdoanh/LayoutGeneration/outputs_detection/14653_test"
    
    print(f"\n{'='*70}")
    print(f"Testing object detection on: {keyframes_folder}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
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
    
    print(f"\n{'='*70}")
    print("✅ Detection completed successfully!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - detection_results.json")
    print(f"  - image_*_annotated.jpg (with labels)")
    print(f"  - image_*_boxes.jpg (boxes only)")
    
    # Print summary
    if results:
        total_detections = sum(len(r.get("detections", [])) for r in results)
        print(f"\nSummary:")
        print(f"  - Total images processed: {len(results)}")
        print(f"  - Total objects detected: {total_detections}")
        print(f"  - Average objects per image: {total_detections/len(results):.1f}")

if __name__ == "__main__":
    main()
