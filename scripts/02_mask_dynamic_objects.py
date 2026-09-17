#!/usr/bin/env python3
"""
Stage 2 — Dynamic Object Masking (YOLOv8n)

Detects dynamic objects (people, vehicles, animals) using YOLOv8n and
zeroes out their bounding boxes in each frame. Bounding-box masking only —
no pixel-level segmentation, per CLAUDE.md / VGGT usage guidelines.

Input:
    --input_dir     Directory containing extracted frames (from Stage 1)
    --output_dir    Directory to write masked frames

Output:
    output_dir/frame_NNNN.jpg       Masked frames (dynamic objects blacked out)
    output_dir/../mask_report.json   Per-frame detection summary
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("02_mask_dynamic_objects")

# COCO class IDs for dynamic objects that should be masked before reconstruction.
# These are objects that move between frames and would corrupt the 3D reconstruction.
DYNAMIC_COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: Mask dynamic objects (people/vehicles/animals) using YOLOv8n"
    )
    parser.add_argument(
        "--input_dir", required=True, type=str,
        help="Directory containing extracted frames (from Stage 1)"
    )
    parser.add_argument(
        "--output_dir", required=True, type=str,
        help="Output directory for masked frames"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="YOLOv8 model path or name (default: yolov8n.pt, auto-downloads)"
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.3,
        help="Detection confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--mask_value", type=int, default=0,
        help="Pixel value to fill masked regions (default: 0 = black)"
    )
    parser.add_argument(
        "--mask_padding", type=int, default=5,
        help="Pixels to pad around each bounding box (default: 5)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for YOLO inference (default: auto — cuda if available, else cpu)"
    )
    parser.add_argument(
        "--report_file", type=str, default=None,
        help="Path for detection report JSON (default: output_dir/../mask_report.json)"
    )
    return parser.parse_args()


def load_yolo_model(model_path: str, device: str = None):
    """
    Load YOLOv8 model. Auto-downloads yolov8n.pt if not present.

    Returns the YOLO model instance.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error(
            "ultralytics package not installed. "
            "Install with: pip install ultralytics"
        )
        sys.exit(1)

    log.info("Loading YOLO model: %s", model_path)
    model = YOLO(model_path)

    if device:
        model.to(device)

    return model


def mask_frame(image: np.ndarray, detections, conf_threshold: float,
               mask_value: int = 0, padding: int = 5) -> tuple:
    """
    Zero out bounding boxes for dynamic objects in a frame.

    Returns:
        masked_image: Image with dynamic object regions zeroed out
        detection_info: List of dicts with detection details
    """
    masked = image.copy()
    h, w = masked.shape[:2]
    detection_info = []

    if detections is None or len(detections) == 0:
        return masked, detection_info

    boxes = detections.boxes
    if boxes is None or len(boxes) == 0:
        return masked, detection_info

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())

        # Only mask dynamic object classes above confidence threshold
        if cls_id not in DYNAMIC_COCO_CLASSES or conf < conf_threshold:
            continue

        # Get bounding box coordinates (xyxy format)
        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

        # Apply padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Zero out the bounding box region
        masked[y1:y2, x1:x2] = mask_value

        detection_info.append({
            "class_id": cls_id,
            "class_name": DYNAMIC_COCO_CLASSES[cls_id],
            "confidence": round(conf, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    return masked, detection_info


def run_masking(args):
    """Main masking pipeline."""

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    frame_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not frame_files:
        log.error("No image files found in %s", input_dir)
        sys.exit(1)

    log.info("Found %d frames to process", len(frame_files))

    # Load YOLO model
    model = load_yolo_model(args.model, args.device)

    # Process each frame
    report = {
        "total_frames": len(frame_files),
        "frames_with_detections": 0,
        "total_detections": 0,
        "class_counts": {},
        "per_frame": [],
    }

    for frame_path in tqdm(frame_files, desc="Masking frames", unit="frame"):
        image = cv2.imread(str(frame_path))
        if image is None:
            log.warning("Cannot read image: %s — skipping", frame_path)
            continue

        # Run YOLO inference
        results = model(image, verbose=False, conf=args.conf_threshold)

        # Mask dynamic objects
        masked_image, detections = mask_frame(
            image, results[0], args.conf_threshold,
            args.mask_value, args.mask_padding,
        )

        # Save masked frame (same filename)
        output_path = output_dir / frame_path.name
        cv2.imwrite(str(output_path), masked_image)

        # Update report
        frame_report = {
            "frame": frame_path.name,
            "num_detections": len(detections),
            "detections": detections,
        }
        report["per_frame"].append(frame_report)

        if detections:
            report["frames_with_detections"] += 1
            report["total_detections"] += len(detections)
            for det in detections:
                cls_name = det["class_name"]
                report["class_counts"][cls_name] = (
                    report["class_counts"].get(cls_name, 0) + 1
                )

    # Write report
    report_path = args.report_file
    if report_path is None:
        report_path = str(output_dir.parent / "mask_report.json")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Log summary
    log.info(
        "Masking complete: %d/%d frames had dynamic objects",
        report["frames_with_detections"],
        report["total_frames"],
    )
    log.info("Total detections: %d", report["total_detections"])
    if report["class_counts"]:
        log.info("Detection breakdown: %s", report["class_counts"])
    log.info("Report written to %s", report_path)

    return report


if __name__ == "__main__":
    args = parse_args()
    run_masking(args)
    log.info("Stage 2 complete")
