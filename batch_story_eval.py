#!/usr/bin/env python3
"""
Batch Story Coherence Evaluation cho tất cả folders
"""
import os
import sys
import torch
import glob
from objectfree.story_coherence_evaluator import StoryCoherenceEvaluator

def main():
    print("="*70)
    print("BATCH STORY COHERENCE EVALUATION")
    print("="*70)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")
    
    # Khởi tạo evaluator
    print("Initializing Story Coherence Evaluator...")
    evaluator = StoryCoherenceEvaluator(
        blip_model_name="Salesforce/blip-image-captioning-large",
        device=device
    )
    
    # Base folders
    detection_base = "/home/serverai/ltdoanh/LayoutGeneration/outputs_detection"
    keyframes_base = "/home/serverai/ltdoanh/LayoutGeneration/samples/keyframe4check"
    output_base = "/home/serverai/ltdoanh/LayoutGeneration/outputs_story_eval"
    
    # Tìm tất cả detection folders
    detection_folders = []
    for item in sorted(os.listdir(detection_base)):
        item_path = os.path.join(detection_base, item)
        if os.path.isdir(item_path) and item.endswith("_detections"):
            detection_json = os.path.join(item_path, "detection_results.json")
            if os.path.exists(detection_json):
                detection_folders.append((item, detection_json))
    
    if len(detection_folders) == 0:
        print("[ERROR] No detection folders found!")
        return
    
    print(f"\nFound {len(detection_folders)} folders with detection results:")
    for i, (folder, _) in enumerate(detection_folders):
        print(f"  {i+1}. {folder}")
    
    # Process each folder
    all_results = []
    
    for folder_name, detection_json in detection_folders:
        # Extract keyframe folder name (remove "_detections" suffix)
        keyframe_folder_name = folder_name.replace("_detections", "")
        keyframes_folder = os.path.join(keyframes_base, keyframe_folder_name)
        
        if not os.path.exists(keyframes_folder):
            print(f"\n[Skip] Keyframes folder not found: {keyframes_folder}")
            continue
        
        output_dir = os.path.join(output_base, keyframe_folder_name)
        
        print(f"\n{'='*70}")
        print(f"Processing: {keyframe_folder_name}")
        print(f"{'='*70}")
        
        try:
            results = evaluator.evaluate_batch(
                detection_results_json=detection_json,
                keyframes_folder=keyframes_folder,
                output_dir=output_dir,
                save_crops=True
            )
            
            all_results.extend(results)
            print(f"✓ Completed {keyframe_folder_name}: {len(results)} images")
            
        except Exception as e:
            print(f"✗ Error processing {keyframe_folder_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Global summary
    print(f"\n{'='*70}")
    print(f"BATCH EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total folders processed: {len(detection_folders)}")
    print(f"Total images evaluated: {len(all_results)}")
    
    if all_results:
        import numpy as np
        avg_sim = np.mean([r['avg_similarity'] for r in all_results])
        print(f"Overall average similarity: {avg_sim:.3f}")
        
        # Top 5 images by similarity
        sorted_results = sorted(all_results, key=lambda x: x['avg_similarity'], reverse=True)
        print(f"\nTop 5 images by story similarity:")
        for i, r in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {os.path.basename(r['image_path'])}")
            print(f"     Caption: \"{r['full_caption']}\"")
            print(f"     Avg similarity: {r['avg_similarity']:.3f}")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_base}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
