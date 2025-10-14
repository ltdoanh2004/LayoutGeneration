#!/usr/bin/env python3
"""
Batch evaluation of all keyframe folders using comprehensive metrics
"""
import os
import sys
import glob
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(parent_dir)

from objectfree.eval_comprehensive import ImageRegionAnalyzer


def process_single_image(image_path, output_dir):
    """Process a single image and return statistics"""
    try:
        # Create analyzer
        analyzer = ImageRegionAnalyzer(image_path)
        
        # Compute combined score
        combined_score, individual_maps, key_regions = analyzer.visualize_and_save(output_dir)
        
        # Extract statistics
        stats = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'entropy_mean': float(individual_maps['entropy'].mean()),
            'entropy_std': float(individual_maps['entropy'].std()),
            'entropy_max': float(individual_maps['entropy'].max()),
            'edge_mean': float(individual_maps['edge'].mean()),
            'edge_std': float(individual_maps['edge'].std()),
            'edge_max': float(individual_maps['edge'].max()),
            'color_mean': float(individual_maps['color'].mean()),
            'color_std': float(individual_maps['color'].std()),
            'color_max': float(individual_maps['color'].max()),
            'combined_mean': float(combined_score.mean()),
            'combined_std': float(combined_score.std()),
            'combined_max': float(combined_score.max()),
            'num_key_regions': len(key_regions),
            'top_region_score': float(key_regions[0][1]) if key_regions else 0.0,
            'status': 'success'
        }
        
        return stats
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'status': 'failed',
            'error': str(e)
        }


def process_folder(folder_path, output_base_dir):
    """Process all images in a keyframe folder"""
    folder_name = os.path.basename(folder_path)
    print(f"\n{'='*70}")
    print(f"Processing: {folder_name}")
    print(f"{'='*70}")
    
    # Get all image files
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    image_paths = [p for p in image_paths if "preview" not in os.path.basename(p).lower()]
    
    if len(image_paths) == 0:
        print(f"   No images found in {folder_path}")
        return []
    
    print(f"Found {len(image_paths)} images")
    
    # Create output directory for this folder
    output_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    all_stats = []
    for image_path in tqdm(image_paths, desc=f"  Processing {folder_name}"):
        image_name = os.path.basename(image_path)
        image_output_dir = os.path.join(output_dir, image_name.replace('.jpg', ''))
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Process image
        stats = process_single_image(image_path, image_output_dir)
        all_stats.append(stats)
    
    # Save summary for this folder
    summary_path = os.path.join(output_dir, 'folder_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"   Saved: {summary_path}")
    
    # Create CSV summary
    df = pd.DataFrame(all_stats)
    csv_path = os.path.join(output_dir, 'folder_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")
    
    # Print folder statistics
    success_count = sum(1 for s in all_stats if s.get('status') == 'success')
    print(f"\n   Summary for {folder_name}:")
    print(f"   - Total images: {len(all_stats)}")
    print(f"   - Success: {success_count}")
    print(f"   - Failed: {len(all_stats) - success_count}")
    
    if success_count > 0:
        avg_entropy = df[df['status'] == 'success']['entropy_mean'].mean()
        avg_edge = df[df['status'] == 'success']['edge_mean'].mean()
        avg_color = df[df['status'] == 'success']['color_mean'].mean()
        avg_combined = df[df['status'] == 'success']['combined_mean'].mean()
        
        print(f"   - Average Entropy: {avg_entropy:.3f}")
        print(f"   - Average Edge Density: {avg_edge:.3f}")
        print(f"   - Average Color Variance: {avg_color:.3f}")
        print(f"   - Average Combined Score: {avg_combined:.3f}")
    
    return all_stats


def main():
    """Main function to process all keyframe folders"""
    print("="*70)
    print("BATCH COMPREHENSIVE IMAGE EVALUATION")
    print("="*70)
    
    # Base folder containing all keyframe folders
    base_folder = "/home/serverai/ltdoanh/LayoutGeneration/samples/keyframe4check"
    output_base_dir = "/home/serverai/ltdoanh/LayoutGeneration/outputs_eval"
    
    # Get all keyframe folders
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
    
    # Process all folders
    all_results = {}
    for folder_path in keyframe_folders:
        folder_name = os.path.basename(folder_path)
        stats = process_folder(folder_path, output_base_dir)
        all_results[folder_name] = stats
    
    # Create global summary
    print(f"\n{'='*70}")
    print("Creating global summary...")
    print(f"{'='*70}")
    
    # Flatten all stats
    all_stats_flat = []
    for folder_name, stats_list in all_results.items():
        for stats in stats_list:
            stats['folder'] = folder_name
            all_stats_flat.append(stats)
    
    # Save global summary
    global_json_path = os.path.join(output_base_dir, 'global_summary.json')
    with open(global_json_path, 'w') as f:
        json.dump(all_stats_flat, f, indent=2)
    print(f"   Saved: {global_json_path}")
    
    # Create global CSV
    df_global = pd.DataFrame(all_stats_flat)
    global_csv_path = os.path.join(output_base_dir, 'global_summary.csv')
    df_global.to_csv(global_csv_path, index=False)
    print(f"   Saved: {global_csv_path}")
    
    # Print global statistics
    print(f"\n{'='*70}")
    print("GLOBAL STATISTICS")
    print(f"{'='*70}")
    
    total_images = len(all_stats_flat)
    success_images = sum(1 for s in all_stats_flat if s.get('status') == 'success')
    
    print(f"Total folders processed: {len(keyframe_folders)}")
    print(f"Total images processed: {total_images}")
    print(f"Successful: {success_images}")
    print(f"Failed: {total_images - success_images}")
    
    if success_images > 0:
        df_success = df_global[df_global['status'] == 'success']
        
        print(f"\nAverage Metrics (across all images):")
        print(f"  - Entropy Mean: {df_success['entropy_mean'].mean():.3f} ± {df_success['entropy_mean'].std():.3f}")
        print(f"  - Edge Density Mean: {df_success['edge_mean'].mean():.3f} ± {df_success['edge_mean'].std():.3f}")
        print(f"  - Color Variance Mean: {df_success['color_mean'].mean():.3f} ± {df_success['color_mean'].std():.3f}")
        print(f"  - Combined Score Mean: {df_success['combined_mean'].mean():.3f} ± {df_success['combined_mean'].std():.3f}")
        
        print(f"\nTop 5 images by Combined Score:")
        top_images = df_success.nlargest(5, 'combined_mean')[['folder', 'image_name', 'combined_mean', 'entropy_mean', 'edge_mean', 'color_mean']]
        for idx, row in top_images.iterrows():
            print(f"  {row['folder']}/{row['image_name']}")
            print(f"    Combined: {row['combined_mean']:.3f}, Entropy: {row['entropy_mean']:.3f}, Edge: {row['edge_mean']:.3f}, Color: {row['color_mean']:.3f}")
    
    print(f"\n{'='*70}")
    print("BATCH EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_base_dir}/")


if __name__ == "__main__":
    main()
