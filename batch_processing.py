#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch processing script to run pipeline on all MP4 videos in Data folder
"""

import os
import glob
import subprocess
import sys
from pathlib import Path

def find_mp4_videos(data_folder: str) -> list:
    """Find all MP4 files in the data folder"""
    mp4_files = []
    
    # Search for .mp4 files (case insensitive)
    patterns = ['*.mp4', '*.MP4', '*.Mp4', '*.mP4']
    
    for pattern in patterns:
        files = glob.glob(os.path.join(data_folder, pattern))
        mp4_files.extend(files)
    
    # Also search in subdirectories
    for pattern in patterns:
        files = glob.glob(os.path.join(data_folder, '**', pattern), recursive=True)
        mp4_files.extend(files)
    
    # Remove duplicates and sort
    mp4_files = sorted(list(set(mp4_files)))
    return mp4_files

def run_pipeline_for_video(video_path: str, base_output_dir: str = "outputs") -> bool:
    """Run pipeline for a single video"""
    
    # Get video filename without extension for output naming
    video_name = Path(video_path).stem
    
    # Construct the command
    cmd = [
        "python", "pipeline.py",
        "--video", video_path,
        "--backend", "transnetv2",
        "--model_dir", "src/models/TransNetV2",
        "--prob_threshold", "0.7",
        "--distance_backend", "dists",
        "--dists_as_distance", "1",
        "--sample_stride", "12",
        "--max_frames_per_scene", "40",
        "--keyframes_per_scene", "2",
        "--nms_radius", "4",
        "--resize_w", "320",
        "--resize_h", "320",
        "--out_dir", f"{base_output_dir}/run_tv2_dists"
    ]
    
    print(f"\n{'='*60}")
    print(f"Processing: {video_path}")
    print(f"Output will be in: {base_output_dir}/run_tv2_dists_{video_name}")
    print(f"{'='*60}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ SUCCESS: {video_name}")
        print(f"Output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {video_name}")
        print(f"Error: {e.stderr}")
        print(f"Return code: {e.returncode}")
        return False
    
    except Exception as e:
        print(f"❌ ERROR: {video_name}")
        print(f"Unexpected error: {str(e)}")
        return False

def run_eval_and_visualize_for_all_videos(data_folder: str, eval_script: str, visualize_script: str, out_base: str):
    """
    Run eval_keyframes.py and visualize for all videos in data_folder.
    Assumes keyframes and scenes are in outputs/run_tv2_dists_{video_name}/
    """
    mp4_files = find_mp4_videos(data_folder)
    if not mp4_files:
        print(f"No MP4 videos found in {data_folder}")
        return

    for video_path in mp4_files:
        video_name = Path(video_path).stem
        # Đúng với pipeline: outputs/run_tv2_dists_{video_name}/scenes.json
        pipeline_dir = f"outputs/run_tv2_dists_{video_name}"
        scenes_json = f"{pipeline_dir}/scenes.json"
        keyframes_csv = f"{pipeline_dir}/keyframes.csv"
        out_dir = f"{out_base}/eval_{video_name}"

        # Run eval_keyframes.py
        eval_cmd = [
            "python", eval_script,
            "--video", video_path,
            "--scenes_json", scenes_json,
            "--keyframes_csv", keyframes_csv,
            "--out_dir", out_dir,
            "--backbone", "resnet50",
            "--sample_stride", "10",  # match stride in paper
            "--max_frames_eval", "200",
            "--tau", "0.3"
        ]
        print(f"\n=== Evaluating keyframes for {video_name} ===")
        try:
            subprocess.run(eval_cmd, check=True)
            print(f"✅ Eval done for {video_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Eval failed for {video_name}: {e}")

        # Run visualize with proper PYTHONPATH
        viz_cmd = [
            "python", "-m", "eval.visualize.viz_medoids",
            "--video", video_path,
            "--scenes_json", scenes_json,
            "--keyframes_csv", keyframes_csv,
            "--out_dir", out_dir
        ]
        print(f"\n=== Visualizing keyframes for {video_name} ===")
        try:
            # Set PYTHONPATH to current directory for module imports
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()
            subprocess.run(viz_cmd, check=True, env=env)
            print(f"✅ Visualization done for {video_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Visualization failed for {video_name}: {e}")

   

def main():
    """Main function to process all videos"""
    
    # Configuration
    data_folder = "samples"
    output_base = "outputs"
    
    # Check if Data folder exists
    if not os.path.exists(data_folder):
        print(f"❌ Data folder '{data_folder}' not found!")
        sys.exit(1)
    
    # Find all MP4 videos
    print(f"🔍 Searching for MP4 videos in '{data_folder}'...")
    videos = find_mp4_videos(data_folder)
    
    if not videos:
        print(f"❌ No MP4 videos found in '{data_folder}'")
        sys.exit(1)
    
    print(f"📹 Found {len(videos)} video(s):")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video}")
    
    # Confirm before processing
    response = input(f"\n🚀 Process all {len(videos)} videos? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled by user.")
        sys.exit(0)
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Process each video
    success_count = 0
    failed_videos = []
    
    for i, video_path in enumerate(videos, 1):
        print(f"\n📹 Processing video {i}/{len(videos)}")
        
        success = run_pipeline_for_video(video_path, output_base)
        
        if success:
            success_count += 1
        else:
            failed_videos.append(video_path)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎯 BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successful: {success_count}/{len(videos)}")
    print(f"❌ Failed: {len(failed_videos)}/{len(videos)}")
    
    if failed_videos:
        print(f"\n❌ Failed videos:")
        for video in failed_videos:
            print(f"  - {video}")
    
    print(f"\n📁 All outputs saved in: {output_base}/")

if __name__ == "__main__":
    run_eval_and_visualize_for_all_videos(
        data_folder="samples",
        eval_script="eval_keyframes.py",
        visualize_script="eval/visualize/viz_medoids.py",
        out_base="outputs_eval"
    )
    #main()