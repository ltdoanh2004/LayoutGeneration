#!/bin/bash

# Test pipeline with object detection
# Make sure to activate the correct conda environment first: conda activate sam

python pipeline.py \
  --video samples/Sakuga/14653.mp4 \
  --backend pyscenedetect --threshold 27 \
  --distance_backend lpips --lpips_net alex \
  --sample_stride 10 --max_frames_per_scene 30 \
  --keyframes_per_scene 1 --nms_radius 3 \
  --resize_w 320 --resize_h 180 \
  --out_dir outputs/run_14653_with_detection \
  --export_preview \
  --run_object_detection \
  --detection_config objectfree/config.yaml \
  --detection_checkpoint objectfree/Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt
