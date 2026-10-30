"""
Utility functions for image I/O, face cropping, JSON serialization,
and video processing.
"""

import os
import json
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
import cv2
import numpy as np
from PIL import Image
from .detector import DetectionResult, FaceBox


def load_image(source: Union[str, bytes, bytearray, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Load an image from various sources into a standard BGR numpy array.
    
    Args:
        source: filepath, bytes buffer, PIL Image, or numpy array.
        
    Returns:
        np.ndarray in BGR color space.
    """
    if isinstance(source, np.ndarray):
        return source.copy()

    if isinstance(source, Image.Image):
        # PIL Image is RGB -> convert to BGR
        rgb = np.array(source.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(source, (bytes, bytearray)):
        nparr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes.")
        return img

    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Image file not found: {source}")
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"Failed to read image from path: {source}")
        return img

    raise TypeError(f"Unsupported image source type: {type(source)}")


def save_image(image: np.ndarray, output_path: str) -> str:
    """
    Save BGR image array to disk, creating parent folders if necessary.
    """
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"Could not write image to: {output_path}")
    return output_path


def crop_faces(
    image: np.ndarray,
    faces: List[FaceBox],
    margin_ratio: float = 0.2,
) -> List[Dict[str, Any]]:
    """
    Extract individual face avatars from the image with an aesthetic margin.
    
    Args:
        image: BGR numpy array
        faces: list of FaceBox objects
        margin_ratio: fraction of face width/height to add as padding
        
    Returns:
        List of dicts: {"index": int, "image": np.ndarray (BGR), "box": tuple, "score": float}
    """
    h_img, w_img = image.shape[:2]
    crops = []

    for idx, face in enumerate(faces, start=1):
        x, y, w, h = face.x, face.y, face.w, face.h

        # Add proportional padding around the face
        pad_x = int(w * margin_ratio)
        pad_y = int(h * margin_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_img, x + w + pad_x)
        y2 = min(h_img, y + h + pad_y)

        cropped = image[y1:y2, x1:x2]
        if cropped.size > 0:
            crops.append({
                "index": idx,
                "image": cropped,
                "box": (x, y, w, h),
                "crop_coords": (x1, y1, x2, y2),
                "score": face.score,
                "landmarks": face.landmarks,
            })

    return crops


def export_detections_json(
    result: DetectionResult,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export detection results to a dictionary and optionally save to JSON file.
    """
    data = {
        "model_name": result.model_name,
        "total_faces": result.total_faces,
        "inference_time_ms": result.inference_time_ms,
        "image_shape": {
            "height": result.image_shape[0],
            "width": result.image_shape[1],
            "channels": result.image_shape[2],
        },
        "faces": [],
    }

    for idx, face in enumerate(result.faces, start=1):
        face_info = {
            "face_id": idx,
            "bounding_box": {
                "x": face.x,
                "y": face.y,
                "width": face.w,
                "height": face.h,
            },
            "confidence_score": round(float(face.score), 4),
            "landmarks": face.landmarks if face.landmarks else None,
            "sub_features": {
                "eyes": [list(e) for e in face.eyes],
                "smiles": [list(s) for s in face.smiles],
            },
        }
        data["faces"].append(face_info)

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return data


def process_video_file(
    input_video_path: str,
    output_video_path: str,
    detector: Any,
    visualizer: Any,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Process an input video file frame-by-frame and save the annotated video.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    processed_frames = 0
    total_faces_detected = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.detect(frame)
            annotated = visualizer.render(frame, result)
            out.write(annotated)

            processed_frames += 1
            total_faces_detected += result.total_faces

            if progress_callback:
                progress_callback(processed_frames, total_frames)

    finally:
        cap.release()
        out.release()

    return {
        "total_frames": processed_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "output_path": output_video_path,
        "total_face_instances": total_faces_detected,
    }

