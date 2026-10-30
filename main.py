"""
VisionAI - Universal Face Detection CLI & Application Launcher
"""

import argparse
import os
import subprocess
import sys
import cv2

from core.detector import FaceDetector
from core.visualizer import FaceVisualizer
from core.utils import load_image, save_image, crop_faces, export_detections_json, process_video_file


def run_image(args):
    print(f"[VisionAI] Loading image from: {args.image}")
    img = load_image(args.image)

    detector = FaceDetector(
        model_type=args.detector,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
    )
    result = detector.detect(img)

    print(f"\n================ DETECTION RESULTS ================")
    print(f"Model: {result.model_name}")
    print(f"Total Faces Detected: {result.total_faces}")
    print(f"Inference Latency: {result.inference_time_ms} ms")
    print(f"Image Dimensions: {result.image_shape[1]}x{result.image_shape[0]}")
    print("===================================================\n")

    for idx, face in enumerate(result.faces, start=1):
        print(f" * Face #{idx}: BBox (x={face.x}, y={face.y}, w={face.w}, h={face.h}), Score: {face.score:.3f}")
        if face.landmarks:
            print(f"   Landmarks: Eyes={face.landmarks['right_eye']}, {face.landmarks['left_eye']} | Nose={face.landmarks['nose_tip']}")

    # Render
    visualizer = FaceVisualizer(
        theme=args.theme,
        box_style=args.box_style,
        line_thickness=args.thickness,
        privacy_mode=args.privacy,
        blur_strength=args.blur_strength,
    )
    annotated = visualizer.render(img, result)

    # Save output
    out_path = args.output or "output_annotated.jpg"
    save_image(annotated, out_path)
    print(f"\n[VisionAI] Saved annotated image to: {out_path}")

    # Crop faces if requested
    if args.crop:
        crops = crop_faces(img, result.faces)
        crops_dir = os.path.splitext(out_path)[0] + "_faces"
        os.makedirs(crops_dir, exist_ok=True)
        for crop_data in crops:
            cpath = os.path.join(crops_dir, f"face_{crop_data['index']}.jpg")
            save_image(crop_data["image"], cpath)
        print(f"[VisionAI] Saved {len(crops)} cropped face avatar(s) to: {crops_dir}")

    # Export JSON
    if args.export_json:
        export_detections_json(result, args.export_json)
        print(f"[VisionAI] Exported detection metadata to: {args.export_json}")


def run_video(args):
    print(f"[VisionAI] Processing video: {args.video}")
    out_path = args.output or "output_annotated.mp4"

    detector = FaceDetector(
        model_type=args.detector,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
    )
    visualizer = FaceVisualizer(
        theme=args.theme,
        box_style=args.box_style,
        line_thickness=args.thickness,
        privacy_mode=args.privacy,
    )

    def print_progress(curr, total):
        pct = (curr / max(1, total)) * 100
        print(f"\rProgress: [{curr}/{total}] frames ({pct:.1f}%)", end="", flush=True)

    stats = process_video_file(
        input_video_path=args.video,
        output_video_path=out_path,
        detector=detector,
        visualizer=visualizer,
        progress_callback=print_progress,
    )
    print(f"\n\n[VisionAI] Video processing complete! Saved to: {out_path}")
    print(f"Processed {stats['total_frames']} frames with {stats['total_face_instances']} face sightings.")


def run_web(args):
    print("[VisionAI] Launching Streamlit Web Application...")
    app_path = os.path.join(os.path.dirname(__file__), "app_streamlit.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


def run_desktop(args):
    print("[VisionAI] Launching Desktop GUI Application...")
    import app_desktop
    app_desktop.main()


def main():
    parser = argparse.ArgumentParser(
        description="VisionAI - State-of-the-Art Face Detection Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", "-i", type=str, help="Path to input image file")
    group.add_argument("--video", "-v", type=str, help="Path to input video file")
    group.add_argument("--web", action="store_true", help="Launch interactive Streamlit Web UI")
    group.add_argument("--desktop", action="store_true", help="Launch native Tkinter Desktop UI")

    # Options
    parser.add_argument(
        "--detector",
        "-d",
        choices=["yunet", "haar"],
        default="yunet",
        help="Face detector backend: 'yunet' (deep learning ONNX) or 'haar' (classical)",
    )
    parser.add_argument("--conf", "-c", type=float, default=0.6, help="Confidence threshold (0.1 - 1.0)")
    parser.add_argument("--nms", type=float, default=0.3, help="NMS threshold (0.1 - 0.9)")
    parser.add_argument("--theme", default="Cyber Neon", help="Color palette for visualization")
    parser.add_argument(
        "--box-style",
        choices=["corner_hud", "rounded", "rectangle"],
        default="corner_hud",
        help="Bounding box drawing style",
    )
    parser.add_argument("--thickness", type=int, default=2, help="Box line thickness")
    parser.add_argument(
        "--privacy",
        choices=["none", "blur", "mosaic"],
        default="none",
        help="Privacy anonymization filter for detected faces",
    )
    parser.add_argument("--blur-strength", type=int, default=35, help="Blur radius for privacy filter")
    parser.add_argument("--crop", action="store_true", help="Crop and save individual detected faces")
    parser.add_argument("--export-json", type=str, help="Path to save detection metadata JSON")
    parser.add_argument("--output", "-o", type=str, help="Output destination file path")

    args = parser.parse_args()

    if args.web:
        run_web(args)
    elif args.desktop:
        run_desktop(args)
    elif args.video:
        run_video(args)
    elif args.image:
        run_image(args)
    else:
        # Default behavior: run on portrait sample to demonstrate functionality
        sample_path = os.path.join(os.path.dirname(__file__), "samples", "sample_portrait.jpg")
        if os.path.exists(sample_path):
            print("[VisionAI] No arguments specified. Running demonstration on sample image...")
            args.image = sample_path
            args.crop = True
            args.export_json = "detection_output.json"
            run_image(args)
            print("\n[VisionAI] Tip:")
            print("  Run 'python main.py --web' to launch the interactive browser interface.")
            print("  Run 'python main.py --desktop' to launch the desktop application.")
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
