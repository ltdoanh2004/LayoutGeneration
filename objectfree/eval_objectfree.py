#!/usr/bin/env python3
"""
Complete Pipeline: Object Detection → Story Coherence → Comprehensive Evaluation
Input: keyframes folder → Output: all evaluation results
"""
import os
import sys
import torch
import json
import glob
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Import các modules
from inference_dino import LoadDetector
from story_coherence_evaluator import StoryCoherenceEvaluator
from eval_comprehensive import ImageRegionAnalyzer


class CompletePipeline:
    """Complete pipeline combining all evaluation methods"""
    
    def __init__(self, device="cuda", verbose=False):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.verbose = verbose
        if self.verbose:
            print(f"Using device: {self.device}")
        
        # Initialize evaluators
        self.object_detector = None
        self.story_evaluator = None
        self.comprehensive_analyzer = None
    
    def initialize_detectors(self):
        """Initialize all detectors"""
        self.object_detector = LoadDetector(
            config_path="objectfree/config.yaml",
            checkpoint_path="/home/serverai/ltdoanh/LayoutGeneration/objectfree/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt",
            image_path=None,
            device=self.device,
            batch_size=1,
            output_dir=None  # Will be set before use
        )
        
        self.story_evaluator = StoryCoherenceEvaluator(
            blip_model_name="Salesforce/blip-image-captioning-large",
            device=self.device
        )
        
        self.comprehensive_analyzer = ImageRegionAnalyzer  # Class, not instance
    
    def run_object_detection(self, keyframes_folder, output_dir):
        """Step 1: Run object detection"""
        if self.verbose:
            print("STEP 1: OBJECT DETECTION")
            print(f"{'='*70}")
        

        results = self.object_detector.run_inference(
            keyframes_folder=keyframes_folder,
            output_dir=output_dir
        )
        if self.verbose:
            print(f"Object detection completed: {len(results)} images processed")
        return True
 
    
    def run_story_coherence(self, detection_json, keyframes_folder, output_dir):
        """Step 2: Run story coherence evaluation"""
        if self.verbose:
            print("STEP 2: STORY COHERENCE EVALUATION")
            print(f"{'='*70}")
    
        results = self.story_evaluator.evaluate_batch(
            detection_results_json=detection_json,
            keyframes_folder=keyframes_folder,
            output_dir=output_dir,
            save_crops=True
        )
        if self.verbose:
            print(f"Story coherence completed: {len(results)} images evaluated")
        return results
    
    def run_comprehensive_eval(self, keyframes_folder, output_dir):
        """Step 3: Run comprehensive evaluation (entropy, edge, color)"""
        if self.verbose:
            print("STEP 3: COMPREHENSIVE EVALUATION")
            print(f"{'='*70}")
        
        # Get all images
        image_paths = sorted(glob.glob(os.path.join(keyframes_folder, "*.jpg")))
        image_paths = [p for p in image_paths if "preview" not in os.path.basename(p).lower()]
        
        if len(image_paths) == 0:
            if self.verbose:
                print(f"✗ No images found in {keyframes_folder}")
            return []
        
        if self.verbose:
            print(f"Processing {len(image_paths)} images...")
        
        all_stats = []
        for image_path in image_paths:  # Disabled tqdm for cleaner output
            # Create analyzer for each image
            analyzer = ImageRegionAnalyzer(image_path)
            
            # Compute combined score
            combined_score, individual_maps, key_regions = analyzer.visualize_and_save(output_dir)
            
            # Extract stats (assuming visualize_and_save returns the necessary data)
            stats = {
                'image_path': image_path,
                'combined_score': combined_score,
                'status': 'success'
            }
            all_stats.append(stats)
    
        # Save results
        summary_path = os.path.join(output_dir, 'comprehensive_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        df = pd.DataFrame(all_stats)
        csv_path = os.path.join(output_dir, 'comprehensive_summary.csv')
        df.to_csv(csv_path, index=False)
        
        if self.verbose:
            print(f"Comprehensive evaluation completed: {len(all_stats)} images processed")
        return all_stats
    
    def create_final_report(self, folder_name, detection_results, story_results, comprehensive_results, output_dir):
        """Create final combined report"""
        if self.verbose:
            print("CREATING FINAL REPORT")
            print(f"{'='*70}")
        
        # Combine all results
        final_report = {
            'folder_name': folder_name,
            'timestamp': str(pd.Timestamp.now()),
            'summary': {
                'total_images': len(detection_results) if detection_results else 0,
                'story_eval_images': len(story_results),
                'comprehensive_eval_images': len(comprehensive_results),
                'successful_comprehensive': sum(1 for r in comprehensive_results if r.get('status') == 'success')
            },
            'detection_results': detection_results,
            'story_results': story_results,
            'comprehensive_results': comprehensive_results
        }
        
        # Calculate overall metrics
        if comprehensive_results:
            df_comp = pd.DataFrame([r for r in comprehensive_results if r.get('status') == 'success'])
            if not df_comp.empty:
                final_report['overall_metrics'] = {
                    'avg_entropy': df_comp['entropy_mean'].mean() if 'entropy_mean' in df_comp else 0,
                    'avg_edge_density': df_comp['edge_mean'].mean() if 'edge_mean' in df_comp else 0,
                    'avg_color_variance': df_comp['color_mean'].mean() if 'color_mean' in df_comp else 0,
                    'avg_combined_score': df_comp['combined_mean'].mean() if 'combined_mean' in df_comp else 0
                }
        
        # Save final report
        report_path = os.path.join(output_dir, 'final_report.json')
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        if self.verbose:
            print(f" Final report saved: {report_path}")
        return final_report
    
    def process_single_folder(self, keyframes_folder, output_base="/home/serverai/ltdoanh/LayoutGeneration/outputs_complete"):
        """Process a single keyframes folder through all steps"""
        folder_name = os.path.basename(keyframes_folder)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"PROCESSING FOLDER: {folder_name}")
            print(f"{'='*80}")
        
        # Create output directory
        output_dir = os.path.join(output_base, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Object Detection
        detection_output = os.path.join(output_dir, 'object_detection')
        detection_success = self.run_object_detection(keyframes_folder, detection_output)
        
        detection_results = None
        story_results = []
        comprehensive_results = []
        
        if detection_success:
            detection_json = os.path.join(detection_output, 'detection_results.json')
            
            # Step 2: Story Coherence
            story_output = os.path.join(output_dir, 'story_coherence')
            story_results = self.run_story_coherence(detection_json, keyframes_folder, story_output)
            
            # Load detection results for final report
            try:
                with open(detection_json, 'r') as f:
                    detection_data = json.load(f)
                    detection_results = detection_data.get('results', [])
            except:
                detection_results = None
        
        # Step 3: Comprehensive Evaluation (always run, independent of detection)
        comprehensive_output = os.path.join(output_dir, 'comprehensive_eval')
        comprehensive_results = self.run_comprehensive_eval(keyframes_folder, comprehensive_output)
        
        # Create final report
        final_report = self.create_final_report(
            folder_name, detection_results, story_results, comprehensive_results, output_dir
        )
        
        return {
            'folder_name': folder_name,
            'detection_success': detection_success,
            'story_results_count': len(story_results),
            'comprehensive_results_count': len(comprehensive_results),
            'output_dir': output_dir
        }


def main():
    parser = argparse.ArgumentParser(description="Complete Pipeline for Layout Generation Evaluation")
    parser.add_argument('--all', action='store_true', help='Process all keyframe folders')
    parser.add_argument('--output', type=str, default='/home/serverai/ltdoanh/LayoutGeneration/outputs_complete', help='Base output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('folder', nargs='?', help='Single folder to process (if not --all)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CompletePipeline(device=args.device, verbose=args.verbose)
    pipeline.initialize_detectors()
    
    if args.all:
        # Process all folders
        base_folder = "/home/serverai/ltdoanh/LayoutGeneration/samples/keyframe4check"
        keyframe_folders = []
        
        for item in sorted(os.listdir(base_folder)):
            item_path = os.path.join(base_folder, item)
            if os.path.isdir(item_path) and item.endswith("_keyframes"):
                keyframe_folders.append(item_path)
        
        if args.verbose:
            print(f"\nFound {len(keyframe_folders)} folders to process:")
            for i, folder in enumerate(keyframe_folders):
                print(f"  {i+1}. {os.path.basename(folder)}")
        
        # Process all folders
        all_results = []
        for folder_path in keyframe_folders:
            result = pipeline.process_single_folder(folder_path, args.output)
            all_results.append(result)
        
        if args.verbose:
            total_story = sum(r['story_results_count'] for r in all_results)
            total_comprehensive = sum(r['comprehensive_results_count'] for r in all_results)
            print(f"\nCompleted processing {len(all_results)} folders")
            print(f"Total story evaluations: {total_story}")
            print(f"Total comprehensive evaluations: {total_comprehensive}")
    else:
        # Process single folder
        if not args.folder:
            parser.error("Must specify --all or provide a folder path")
        keyframes_folder = args.folder
        if not os.path.exists(keyframes_folder):
            print(f"[ERROR] Folder not found: {keyframes_folder}")
            return
        
        result = pipeline.process_single_folder(keyframes_folder, args.output)
        if args.verbose:
            print(f"Processed folder: {result['folder_name']}")


if __name__ == "__main__":
    main()