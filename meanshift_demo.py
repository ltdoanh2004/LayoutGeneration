#!/usr/bin/env python3
"""
Visualize Mean Shift advantage on selective expansion
Shows direction-aware growth/shrinking
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import torch

from objectfree.bbox_refinement import BBoxRefinement
from objectfree.inference_dino import LoadDetector


def main():
    """Show mean shift advantage with real SAM2 detections"""
    
    img_path = '/home/serverai/ltdoanh/LayoutGeneration/data/samples/keyframe4check/21724_keyframes/0001_clip06_frame059_21724_6.jpg'

    image = cv2.imread(img_path)
    h, w = image.shape[:2]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("\n" + "="*80)
    print("MEAN SHIFT ADVANTAGE: Real SAM2 Detections")
    print("="*80)
    print(f"Image: {Path(img_path).name} ({w}x{h})")
    print("Loading SAM2 detector...")
    
    # Initialize SAM2 detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = LoadDetector(
        config_path="objectfree/config.yaml",
        checkpoint_path="./objectfree/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt",
        image_path=[img_path],
        device=device,
        batch_size=1,
        output_dir="meanshift_demo/detector_output"
    )
    
    # Run detection
    print("Running SAM2 detection...")
    results = detector()
    
    if not results or "detections" not in results[0]:
        print("[ERROR] No detections from SAM2")
        return
    
    raw_detections = results[0]["detections"]
    print(f"✓ Detected {len(raw_detections)} bboxes from SAM2")
    
    # Get top 5 detections by confidence
    sorted_dets = sorted(raw_detections, key=lambda x: x.get('score', 0), reverse=True)[:5]
    test_bboxes = [det['bbox'] for det in sorted_dets]
    
    print(f"Using top {len(test_bboxes)} detections for comparison")
    print("\nMean Shift: Co/giãn theo hướng - co vùng ít quan trọng, giãn vùng quan trọng")
    
    refiner = BBoxRefinement()
    output_dir = "meanshift_demo"
    Path(output_dir).mkdir(exist_ok=True)
    
    for bbox_idx, bbox in enumerate(test_bboxes):
        print(f"\n{'─'*80}")
        print(f"BBox {bbox_idx}: {bbox}")
        print(f"Original size: {bbox[2]-bbox[0]}x{bbox[3]-bbox[1]}")
        print(f"{'─'*80}")
        
        # Run refinement strategies
        results = {}
        
        # Standard approach (no mean shift)
        refined_std, heatmap_std, info_std = refiner.refine_bbox(
            image, bbox,
            use_mean_shift=False,
            distance_weight=True,
            use_kde=True,
        )
        results['Standard'] = (refined_std, heatmap_std, info_std)
        
        # Mean shift approach
        refined_ms, heatmap_ms, info_ms = refiner.refine_bbox(
            image, bbox,
            use_mean_shift=True,
            distance_weight=True,
            use_kde=True,
        )
        results['Mean Shift'] = (refined_ms, heatmap_ms, info_ms)
        
        # Compare sizes
        for name, (refined, heatmap, info) in results.items():
            x1, y1, x2, y2 = [int(c) for c in refined]
            size = (x2 - x1) * (y2 - y1)
            print(f"\n  [{name}]")
            print(f"    Refined: {refined}")
            print(f"    Size: {x2-x1}x{y2-y1} (expansion: {info['expansion_ratio']:.2f}x)")
            print(f"    Area: {size} pixels")
        
        # Create comparison visualization
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"BBox {bbox_idx}: Mean Shift vs Standard", 
                    fontsize=16, fontweight='bold')
        
        strategies = [
            ('Standard (KDE+Distance)', results['Standard']),
            ('Mean Shift + Distance', results['Mean Shift']),
        ]
        
        for col, (name, (refined, heatmap, info)) in enumerate(strategies):
            # Original + refined bbox
            ax1 = plt.subplot(2, 2, col*2 + 1)
            img_viz = img_rgb.copy()
            
            # Original bbox (red)
            ox1, oy1, ox2, oy2 = [int(c) for c in bbox]
            cv2.rectangle(img_viz, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
            
            # Refined bbox (green)
            rx1, ry1, rx2, ry2 = [int(c) for c in refined]
            cv2.rectangle(img_viz, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
            
            ax1.imshow(img_viz)
            title = f"{name}\nExpansion: {info['expansion_ratio']:.2f}x"
            ax1.set_title(title, fontweight='bold', fontsize=12)
            ax1.axis('off')
            
            # Heatmap
            ax2 = plt.subplot(2, 2, col*2 + 2)
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            ax2.imshow(heatmap_rgb)
            
            info_text = f"KDE Thresh: {info['kde_threshold']:.3f}\nPixels: {info['important_pixel_count']}"
            ax2.set_title(info_text, fontweight='bold', fontsize=11)
            ax2.axis('off')
        
        plt.tight_layout()
        
        viz_file = f"{output_dir}/meanshift_demo_bbox{bbox_idx}.png"
        plt.savefig(viz_file, dpi=100, bbox_inches='tight')
        print(f"\n  💾 Saved: {viz_file}")
        plt.close()
    
    print(f"\n{'='*80}")
    print("KEY DIFFERENCES: Mean Shift vs Standard")
    print("─"*80)
    print("Standard (KDE + Distance):")
    print("  • Uniform growth in all directions")
    print("  • Gaussian decay from center")
    print("  • Result: Symmetric expansion")
    print()
    print("Mean Shift:")
    print("  • Clustering-based growth detection")
    print("  • Expand toward high-importance clusters")
    print("  • Shrink from low-importance areas")
    print("  • Result: Asymmetric, intelligent expansion ✓")
    print("="*80 + "\n")
    
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
