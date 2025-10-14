#!/usr/bin/env python3
"""
Test Story Coherence Evaluation với detection results
"""
import os
import sys
import torch
from objectfree.story_coherence_evaluator import StoryCoherenceEvaluator

def main():
    print("="*70)
    print("STORY COHERENCE EVALUATION TEST")
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
    
    # Test với detection results đã có
    detection_json = "/home/serverai/ltdoanh/LayoutGeneration/outputs_detection/6261_keyframes_detections/detection_results.json"
    keyframes_folder = "/home/serverai/ltdoanh/LayoutGeneration/samples/keyframe4check/6261_keyframes"
    output_dir = "/home/serverai/ltdoanh/LayoutGeneration/outputs_story_eval/6261_test"
    
    if not os.path.exists(detection_json):
        print(f"\n[ERROR] Detection results not found: {detection_json}")
        print("Please run object detection first!")
        return
    
    if not os.path.exists(keyframes_folder):
        print(f"\n[ERROR] Keyframes folder not found: {keyframes_folder}")
        return
    
    print(f"\n{'='*70}")
    print(f"Detection JSON: {detection_json}")
    print(f"Keyframes folder: {keyframes_folder}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Chạy evaluation
    results = evaluator.evaluate_batch(
        detection_results_json=detection_json,
        keyframes_folder=keyframes_folder,
        output_dir=output_dir,
        save_crops=True
    )
    
    # In summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for i, res in enumerate(results[:3]):  # Show first 3
        print(f"\n{i+1}. {os.path.basename(res['image_path'])}")
        print(f"   Full caption: \"{res['full_caption']}\"")
        print(f"   Crops: {res['num_valid_crops']} | Avg similarity: {res['avg_similarity']:.3f}")
        if res['crops']:
            best_crop = res['crops'][0]  # Đã sort theo similarity
            print(f"   Best crop: \"{best_crop['crop_caption']}\" (sim={best_crop['story_similarity']:.3f})")
    
    if len(results) > 3:
        print(f"\n... and {len(results) - 3} more images")
    
    print(f"\n{'='*70}")
    print(f"✓ Story coherence evaluation completed!")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
