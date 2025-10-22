import os
import cv2
import numpy as np
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv
import argparse

SYSTEM_PROMPT = """You are KeyframeExtractorGPT.
Analyze multiple video frames and select the most representative keyframes for scene changes or important transitions.
Output ONLY valid JSON array with fields:
- "index": frame index
- "timestamp": seconds
- "image": base64 of frame image (optional)
Do NOT include markdown, explanations, or any text outside JSON.
"""

PROMPT = """You are given a batch of consecutive video frames.
Select the frames that are true keyframes representing scene or action changes.
Return ONLY a JSON array with index, timestamp, and optionally image base64.
"""

def detect_scene_changes(video_path, threshold=30.0):
    """Detect scene changes based on frame difference."""
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    frame_indices = []
    timestamps = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            score = np.mean(diff)
            if score > threshold:
                frame_indices.append(idx)
                timestamps.append(idx / fps)
        prev_gray = gray
        idx += 1
    cap.release()
    return frame_indices, timestamps

def frames_to_base64(video_path, frame_indices):
    """Convert selected frames to base64."""
    cap = cv2.VideoCapture(video_path)
    b64_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(buf).decode("ascii")
        b64_frames.append(b64)
    cap.release()
    return b64_frames

def clean_gpt_json(raw_text):
    """Remove code fences or extra text from GPT output before parsing JSON."""
    s = raw_text.strip()
    if s.startswith("```"):
        s = '\n'.join(line for line in s.split('\n') if not line.strip().startswith("```"))
    return s.strip()

def save_keyframes(video_path: str, keyframes: list, output_dir: str = "keyframes"):
    """Lưu các keyframe ra file JPG, đảm bảo mỗi frame index là duy nhất và đúng vị trí trong video."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Lọc duplicate theo index và sắp xếp
    unique = {}
    for kf in keyframes:
        frame_idx = kf.get("index")
        if frame_idx is None:
            timestamp = kf.get("timestamp", 0.0)
            frame_idx = int(timestamp * fps)
        if frame_idx not in unique:
            unique[frame_idx] = kf

    sorted_indices = sorted(unique.keys())

    for idx in sorted_indices:
        kf = unique[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Không đọc được frame {idx}")
            continue
        filename = f"frame_{idx:04d}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, frame)
        print(f"✅ Lưu frame {idx} → {path}")

    cap.release()
    print(f"\n🎞 Hoàn tất lưu {len(sorted_indices)} keyframe(s) vào '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to .mp4 video")
    parser.add_argument("--model", default="gpt-4o", help="Vision-capable model")
    parser.add_argument("--threshold", type=float, default=30.0, help="Scene change threshold")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of frames per GPT batch")
    parser.add_argument("--output_dir", "-o", default="keyframes", help="Directory to save keyframes")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_KEY or OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    print("🎬 Detecting scene changes...")
    frame_indices, timestamps = detect_scene_changes(args.video, threshold=args.threshold)
    print(f"Detected {len(frame_indices)} candidate scene frames.")

    # Convert frames to base64
    b64_frames = frames_to_base64(args.video, frame_indices)

    # Send frames in batches to GPT
    keyframes = []
    for i in range(0, len(b64_frames), args.batch_size):
        batch = b64_frames[i:i + args.batch_size]
        batch_indices = frame_indices[i:i + args.batch_size]
        batch_timestamps = timestamps[i:i + args.batch_size]

        input_content = [{"type": "input_text", "text": PROMPT}]
        for idx, img_b64 in enumerate(batch):
            input_content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"})

        resp = client.responses.create(
            model=args.model,
            max_output_tokens=2048,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": input_content},
            ],
        )

        # Clean and parse JSON
        try:
            print("📝 Raw GPT output:\n", resp.output_text)
            clean_text = clean_gpt_json(resp.output_text)
            batch_keyframes = json.loads(clean_text)
            keyframes.extend(batch_keyframes)
        except Exception as e:
            print(f"⚠️ Failed to parse GPT output: {e}")
            continue

    # Save keyframes JSON
    out_json = "keyframes_gpt.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(keyframes, f, indent=4)
    print(f"\n✅ Keyframes JSON saved to {out_json}")

    # Save keyframes images
    save_keyframes(args.video, keyframes, args.output_dir)

if __name__ == "__main__":
    main()
