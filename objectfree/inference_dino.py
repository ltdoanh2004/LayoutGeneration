import sys

import glob
import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import yaml
import torch.nn as nn
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LoadDetector(nn.Module):
    def __init__(self, config_path, checkpoint_path, image_path, device, batch_size=1, output_dir="outputs"):
        super(LoadDetector, self).__init__()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.image_path = image_path
        self.device = device
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Note: build_sam2 will be called later with correct config from Read_config
        self.predictor = None

    def load_batch_image(self):
        """
        Return a list of PIL.Image objects of length self.batch_size.
        - Accepts self.image_path as a single path (str) or an iterable of paths.
        - Raises FileNotFoundError if any referenced file is missing.
        - If fewer paths than batch_size, repeats the last image to fill the batch.
        """
        # normalize to list of paths
        if isinstance(self.image_path, str):
            paths = [self.image_path]
        else:
            paths = list(self.image_path)

        if len(paths) == 0:
            raise ValueError("No image paths provided in self.image_path")

        images = []
        for p in paths[: self.batch_size]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image file not found: {p}")
            images.append(Image.open(p).convert("RGB"))
        return images
    
    def Read_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.text_prompt = config['config']['TEXT_PROMPT']
        self.sam2_checkpoint = config['config']['SAM2_CHECKPOINT']
        self.sam2_model_config = config['config']['SAM2_MODEL_CONFIG']
        self.grounding_dino_model = config['config']['GROUNDING_DINO_MODEL']
        self.box_threshold = config['config']['BOX_THRESHOLD']
        self.text_threshold = config['config']['TEXT_THRESHOLD']
        return self.text_prompt, self.sam2_checkpoint, self.sam2_model_config, self.grounding_dino_model, self.box_threshold, self.text_threshold
    
    def load_grounding_dino(self):
        """Load the Grounding DINO and SAM2 models."""
        # Load SAM2 with correct argument order: config_file, ckpt_path, device
        print(f"  Loading SAM2 from: {self.sam2_checkpoint}")
        sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("  SAM2 loaded")

        # Load Grounding DINO from HuggingFace
        print(f"  Loading Grounding DINO from HuggingFace: {self.grounding_dino_model}")
        processor = AutoProcessor.from_pretrained(self.grounding_dino_model)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_dino_model).to(self.device)
        print("  Grounding DINO loaded")
        return processor, grounding_model
 
    def _bboxes_to_list(self, boxes_xyxy, scores):
        """
        Convert numpy arrays to list of dicts: {'bbox': [x1,y1,x2,y2], 'score': float}
        """
        results = []
        for (x1, y1, x2, y2), s in zip(boxes_xyxy.tolist(), scores.tolist()):
            results.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "score": float(s)})
        return results

    def forward(self):
        # Load config
        self.text_prompt, self.sam2_checkpoint, self.sam2_model_config, self.grounding_dino_model, self.box_threshold, self.text_threshold = self.Read_config()
        print("Config loaded")

        # Load models
        processor, grounding_model = self.load_grounding_dino()

        # Load images
        images = self.load_batch_image()
        print(f"Loaded {len(images)} images for detection")

        all_results = []  # collect results for JSON

        for i, image in enumerate(images):
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            img_cv = image_bgr.copy()

            # Grounding DINO inference
            inputs = processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            # Post-process with new API
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                target_sizes=[image.size[::-1]]  # (height, width)
            )
            
            # Extract boxes, labels, scores
            boxes_filt = results[0]["boxes"].cpu().numpy()
            labels = results[0]["labels"]
            logits_filt = results[0]["scores"].cpu().numpy()

            if len(boxes_filt) == 0:
                print(f"No objects detected in image {i}")
                # still record empty result
                all_results.append({
                    "image_index": i,
                    "image_size": {"width": image.width, "height": image.height},
                    "detections": []
                })
                continue

            # Boxes are already in xyxy format from post_process
            boxes_xyxy = boxes_filt

            # Prepare detections structure for supervision and JSON
            detections = sv.Detections(
                xyxy=boxes_xyxy,
                confidence=logits_filt,
                class_id=np.zeros(len(boxes_xyxy), dtype=int),
            )

            # Annotate images - use latest supervision API
            box_annotator = sv.BoxAnnotator(thickness=2)
            annotated_frame = box_annotator.annotate(scene=img_cv.copy(), detections=detections)

            # Prepare label strings: use detected labels
            label_list = [f"{label} {s:.2f}" for label, s in zip(labels, logits_filt)]
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
            annotated_image = label_annotator.annotate(annotated_frame, detections=detections, labels=label_list)

            # Save annotated image and boxes-only image
            base_name = f"image_{i}"
            annotated_path = os.path.join(self.output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_image)
            boxes_only = box_annotator.annotate(scene=img_cv.copy(), detections=detections)
            boxes_path = os.path.join(self.output_dir, f"{base_name}_boxes.jpg")
            cv2.imwrite(boxes_path, boxes_only)
            print(f"Saved: {annotated_path}, {boxes_path}")

            # Collect JSON-friendly detection records
            det_list = self._bboxes_to_list(boxes_xyxy, logits_filt)
            all_results.append({
                "image_index": i,
                "image_size": {"width": image.width, "height": image.height},
                "annotated_image": os.path.abspath(annotated_path),
                "boxes_image": os.path.abspath(boxes_path),
                "detections": det_list
            })

        # Save all detections to JSON
        json_path = os.path.join(self.output_dir, "detection_results.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump({"results": all_results, "text_prompt": self.text_prompt}, jf, ensure_ascii=False, indent=2)

        print(f"All detection results saved to: {json_path}")
        return all_results
    
  
    def run_inference(self, keyframes_folder, output_dir):
        """
        Run the object detection pipeline on images in the specified folder.
        """
        image_paths = sorted(glob.glob(os.path.join(keyframes_folder, "*.jpg")))
        image_paths = [p for p in image_paths if "preview" not in os.path.basename(p).lower()]
        
        if len(image_paths) == 0:
            print(f"No images found in {keyframes_folder}")
            return
        
        print(f"Found {len(image_paths)} keyframes to process:")
        for i, p in enumerate(image_paths[:5]):  # show first 5
            print(f"  {i}: {os.path.basename(p)}")
        if len(image_paths) > 5:
            print(f"  ... and {len(image_paths) - 5} more")
        
        # Tạo output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Cập nhật đường dẫn ảnh và output directory
        self.image_path = image_paths
        self.output_dir = output_dir
        self.batch_size = len(image_paths)  # xử lý tất cả ảnh
        
        print("\n" + "="*60)
        print("Starting inference...")
        print("="*60 + "\n")
        
        results = self.forward()
        
        print("\n" + "="*60)
        print("Detection completed!")
        print("="*60)
        print(f"Total images processed: {len(results)}")
        print(f"Results saved in: {output_dir}/")
        print(f"JSON results: {output_dir}/detection_results.json")
        print("\nAnnotated images:")
        for i in range(min(3, len(results))):
            if results[i].get("annotated_image"):
                print(f"  - {results[i]['annotated_image']}")
        if len(results) > 3:
            print(f"  ... and {len(results) - 3} more")
