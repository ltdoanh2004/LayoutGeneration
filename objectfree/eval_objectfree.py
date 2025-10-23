"""
Complete Pipeline: Object Detection -> Story Coherence -> Comprehensive Evaluation
Input: keyframes folder -> Output: all evaluation results
"""
import os
import sys
import torch
import json
import glob
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging
import sys

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Add parent directory to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./objectfree/eval_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



# Import các modules
from inference_dino import LoadDetector
from story_coherence_evaluator import StoryCoherenceEvaluator
from eval_comprehensive import ImageRegionAnalyzer


class GetMeanRegion:
    """Complete get mean region for layout generation"""
    
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize evaluators
        self.object_detector = None
        self.story_evaluator = None
        self.comprehensive_analyzer = None
    
    def initialize_detectors(self, config_path="./objectfree/config.yaml"):
        """Initialize all detectors"""
        self.logger.info("Initializing detectors...")
        self.object_detector = LoadDetector(
            config_path=config_path,
            checkpoint_path="./objectfree/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt",
            image_path=None,
            device=self.device,
            batch_size=1,
            output_dir=None 
        )
        
        self.story_evaluator = StoryCoherenceEvaluator(
            blip_model_name="Salesforce/blip-image-captioning-large",
            device=self.device
        )
        
        self.comprehensive_analyzer = ImageRegionAnalyzer 
        self.logger.info("All detectors initialized successfully")

    def run_object_detection(self, keyframes_folder, output_dir):
        """Step 1: Run object detection"""
        self.logger.info(f"Using keyframes folder: {keyframes_folder}")
        self.logger.info(f"Using output directory: {output_dir}")
        

        results = self.object_detector.run_inference(
            keyframes_folder=keyframes_folder,
            output_dir=output_dir
        )
        return True
 
    
    def run_story_coherence(self, detection_json, keyframes_folder, output_dir):
        """Step 2: Run story coherence evaluation"""
    
        results = self.story_evaluator.evaluate_batch(
            detection_results_json=detection_json,
            keyframes_folder=keyframes_folder,
            output_dir=output_dir,
            save_crops=True
        )
        return results
    
    def run_comprehensive_eval(self, keyframes_folder, output_dir):
        """Step 3: Run comprehensive evaluation (entropy, edge, color)"""
        # Get all images
        image_paths = sorted(glob.glob(os.path.join(keyframes_folder, "*.jpg")))
        image_paths = [p for p in image_paths if "preview" not in os.path.basename(p).lower()]
        
        if len(image_paths) == 0:
            print(f"No images found in {keyframes_folder}")
            return []
        
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
            json.dump(convert_numpy_types(all_stats), f, indent=2)
        
        df = pd.DataFrame(all_stats)
        csv_path = os.path.join(output_dir, 'comprehensive_summary.csv')
        df.to_csv(csv_path, index=False)

        return all_stats
    
    def create_final_report(self, folder_name, detection_results, story_results, comprehensive_results, output_dir):
        """Create final combined report"""
        
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
            json.dump(convert_numpy_types(final_report), f, indent=2)
        return final_report
    
    def process_single_folder(self, keyframes_folder, output_base="/home/serverai/ltdoanh/LayoutGeneration/outputs_complete"):
        """Process a single keyframes folder through all steps"""
        folder_name = os.path.basename(keyframes_folder)
 
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
    parser.add_argument('--input', type=str, required=True, help='Path to keyframes folder to process')
    parser.add_argument('--output', type=str, default='/home/serverai/ltdoanh/LayoutGeneration/outputs_complete', help='Base output directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--config', type=str, default='./objectfree/config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = GetMeanRegion(device=args.device)
    pipeline.initialize_detectors(config_path=args.config)
    result = pipeline.process_single_folder(args.input, args.output)

    logging.info(f"Processing completed for folder: {result['folder_name']} and save in: {result['output_dir']}")

if __name__ == "__main__":
    main()