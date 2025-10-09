# Grounded-SAM-2 Keyframes Detection

## Quick Start

### 1. Activate environment
```bash
conda activate sam
cd /home/serverai/ltdoanh/LayoutGeneration
```

### 2. Run inference on keyframes
```bash
python run_inference_keyframes.py
```

### 3. Check results
```bash
# View output directory
ls -lh outputs_14653/

# Read JSON results
cat outputs_14653/detection_results.json | head -100

# View annotated images (if GUI available)
eog outputs_14653/image_0_annotated.jpg
```

## Files

- **run_inference_keyframes.py**: Main script to run detection on all keyframes
- **config.yaml**: Configuration (text prompts, thresholds, model paths)
- **inference_dino.py**: Detection class implementation
- **14653_keyframes/**: Input keyframes folder
- **outputs_14653/**: Output folder (created automatically)

## Configuration

Edit `config.yaml` to adjust:

```yaml
TEXT_PROMPT: "all objects"  # Add more classes with dots
BOX_THRESHOLD: 0.01  # Lower = more detections (more false positives)
TEXT_THRESHOLD: 0.01  # Text matching threshold
```

## Output Structure

```
outputs_14653/
├── image_0_annotated.jpg    # Keyframe 0 with bbox + labels
├── image_0_boxes.jpg         # Keyframe 0 with bbox only
├── image_1_annotated.jpg
├── image_1_boxes.jpg
├── ...
└── detection_results.json    # All detections in JSON format
```

## JSON Format

```json
{
  "results": [
    {
      "image_index": 0,
      "image_size": {"width": 1920, "height": 1080},
      "annotated_image": "/full/path/to/image_0_annotated.jpg",
      "boxes_image": "/full/path/to/image_0_boxes.jpg",
      "detections": [
        {"bbox": [x1, y1, x2, y2], "score": 0.92}
      ]
    }
  ],
  "text_prompt": "person . car . ..."
}
```

## Troubleshooting

**CUDA OOM (Out of Memory):**
- Reduce batch_size in `run_inference_keyframes.py`
- Use smaller model (already using tiny)

**No detections:**
- Lower BOX_THRESHOLD in `config.yaml` (e.g., 0.20 or 0.15)
- Check TEXT_PROMPT matches objects in images

**Slow inference:**
- Check GPU usage: `nvidia-smi`
- Ensure CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
