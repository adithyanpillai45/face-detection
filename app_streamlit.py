"""
VisionAI - Modern Face Detection & Analysis Studio
Interactive Streamlit Web Application
"""

import io
import json
import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.detector import FaceDetector
from core.visualizer import FaceVisualizer, THEME_COLORS
from core.utils import load_image, crop_faces, export_detections_json, process_video_file

# Streamlit Page Config
st.set_page_config(
    page_title="VisionAI • Face Detection Studio",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00f2fe;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_detector(model_type, conf, nms):
    """Cached or newly configured FaceDetector instance."""
    return FaceDetector(
        model_type=model_type,
        conf_threshold=conf,
        nms_threshold=nms,
    )


def bgr_to_pil(bgr_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array to PIL Image."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def main():
    # Header
    col_t1, col_t2 = st.columns([4, 1])
    with col_t1:
        st.title("👤 VisionAI Face Studio")
        st.caption("High-performance Face Detection, Facial Landmarks, and Privacy Anonymization")
    with col_t2:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 15px;">
                <span style="background: rgba(0, 242, 254, 0.15); color: #00f2fe; padding: 6px 12px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; border: 1px solid #00f2fe;">
                    ⚡ ONNX & Classical CV
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Sidebar Controls
    st.sidebar.header("⚙️ Detector Configuration")

    model_type = st.sidebar.selectbox(
        "Detection Model",
        options=["yunet", "haar"],
        format_func=lambda x: "🧠 YuNet Deep Learning (Recommended)" if x == "yunet" else "⚡ Haar Cascade (Classic)",
        index=0,
    )

    if model_type == "yunet":
        conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05)
        nms_thresh = st.sidebar.slider("NMS Threshold", 0.1, 0.9, 0.3, 0.05)
    else:
        conf_thresh = 0.5
        nms_thresh = 0.3
        st.sidebar.info("Haar cascade uses multi-scale feature filters.")

    st.sidebar.markdown("---")
    st.sidebar.header("🎨 Visual Styling & FX")

    theme_name = st.sidebar.selectbox("Color Palette", list(THEME_COLORS.keys()), index=0)
    box_style = st.sidebar.selectbox(
        "Bounding Box Style",
        options=["corner_hud", "rounded", "rectangle"],
        format_func=lambda x: {
            "corner_hud": "Futuristic HUD Corners",
            "rounded": "Smooth Rounded Box",
            "rectangle": "Classic Rectangle",
        }[x],
        index=0,
    )
    line_thickness = st.sidebar.slider("Line Thickness", 1, 5, 2)

    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        show_landmarks = st.checkbox("Show Landmarks", value=(model_type == "yunet"))
        show_scores = st.checkbox("Show Scores", value=True)
    with col_s2:
        show_hud = st.checkbox("Top HUD Bar", value=True)
        dim_bg = st.checkbox("Dim Background", value=False)

    st.sidebar.markdown("---")
    st.sidebar.header("🛡️ Privacy & Anonymization")
    privacy_mode = st.sidebar.radio("Privacy Filter", ["none", "blur", "mosaic"], horizontal=True)

    blur_strength = 35
    mosaic_size = 15
    if privacy_mode == "blur":
        blur_strength = st.sidebar.slider("Blur Radius", 5, 99, 35, step=2)
    elif privacy_mode == "mosaic":
        mosaic_size = st.sidebar.slider("Pixel Block Size", 4, 40, 15)

    # Initialize Engine
    detector = get_detector(model_type, conf_thresh, nms_thresh)
    visualizer = FaceVisualizer(
        theme=theme_name,
        box_style=box_style,
        line_thickness=line_thickness,
        draw_landmarks=show_landmarks,
        draw_scores=show_scores,
        draw_hud_banner=show_hud,
        privacy_mode=privacy_mode,
        blur_strength=blur_strength,
        mosaic_block_size=mosaic_size,
        dim_background=dim_bg,
    )

    # Tab navigation
    tab_image, tab_camera, tab_video = st.tabs(["🖼️ Image Processing", "📷 Live Camera", "🎥 Video File"])

    # ================= TAB 1: IMAGE PROCESSING =================
    with tab_image:
        col_src, col_opts = st.columns([2, 1])
        with col_src:
            source_type = st.radio(
                "Image Source",
                ["Upload Image", "Sample Presets"],
                horizontal=True,
            )

        input_bgr = None

        if source_type == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a photo (JPG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                input_bgr = load_image(file_bytes)
        else:
            sample_dir = os.path.join(os.path.dirname(__file__), "samples")
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith((".jpg", ".png"))] if os.path.exists(sample_dir) else []
            if sample_files:
                selected_sample = st.selectbox("Choose a sample image", sample_files)
                sample_path = os.path.join(sample_dir, selected_sample)
                input_bgr = load_image(sample_path)
            else:
                st.warning("No sample images found in samples/ directory.")

        if input_bgr is not None:
            # Run detection
            result = detector.detect(input_bgr)
            annotated_bgr = visualizer.render(input_bgr, result)
            annotated_pil = bgr_to_pil(annotated_bgr)

            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value">{result.total_faces}</div>
                        <div class="metric-label">Detected Faces</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value">{result.inference_time_ms} ms</div>
                        <div class="metric-label">Inference Latency</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m3:
                fps = round(1000.0 / result.inference_time_ms, 1) if result.inference_time_ms > 0 else 0
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value">{fps}</div>
                        <div class="metric-label">Theoretical FPS</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-value">{result.image_shape[1]}×{result.image_shape[0]}</div>
                        <div class="metric-label">Resolution</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.write("")
            col_orig, col_annot = st.columns(2)
            with col_orig:
                st.subheader("Original Image")
                st.image(bgr_to_pil(input_bgr), use_container_width=True)
            with col_annot:
                st.subheader("Annotated Output")
                st.image(annotated_pil, use_container_width=True)

            # Action Buttons
            buf = io.BytesIO()
            annotated_pil.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="📥 Download Annotated Image",
                data=buf.getvalue(),
                file_name="visionai_detected_face.jpg",
                mime="image/jpeg",
            )

            # Face Cropping & Gallery
            if result.faces:
                st.markdown("---")
                st.subheader(f"👥 Extracted Faces Gallery ({result.total_faces})")
                face_crops = crop_faces(input_bgr, result.faces)

                cols = st.columns(min(5, max(1, len(face_crops))))
                for i, crop_data in enumerate(face_crops):
                    with cols[i % len(cols)]:
                        crop_pil = bgr_to_pil(crop_data["image"])
                        st.image(crop_pil, use_container_width=True)
                        score_label = f"Score: {int(crop_data['score'] * 100)}%" if crop_data['score'] < 1.0 else "Haar Match"
                        st.caption(f"**Face #{crop_data['index']}** • {score_label}")
                        cbuf = io.BytesIO()
                        crop_pil.save(cbuf, format="JPEG")
                        st.download_button(
                            label=f"⬇️ Face #{crop_data['index']}",
                            data=cbuf.getvalue(),
                            file_name=f"face_{crop_data['index']}.jpg",
                            mime="image/jpeg",
                            key=f"dl_face_{crop_data['index']}",
                        )

                # JSON metadata inspection
                with st.expander("📄 View Detection Metadata (JSON)"):
                    json_data = export_detections_json(result)
                    st.json(json_data)
                    st.download_button(
                        label="📥 Download Detection JSON",
                        data=json.dumps(json_data, indent=2),
                        file_name="face_detections.json",
                        mime="application/json",
                    )
        else:
            st.info("👆 Please upload an image or select a sample preset above to begin face detection.")

    # ================= TAB 2: LIVE CAMERA =================
    with tab_camera:
        st.subheader("📷 Web Camera Snapshot & Detection")
        st.caption("Capture a photo using your webcam to detect faces instantly.")

        camera_photo = st.camera_input("Take a photo")

        if camera_photo is not None:
            cam_bytes = camera_photo.read()
            cam_bgr = load_image(cam_bytes)

            cam_result = detector.detect(cam_bgr)
            cam_annotated_bgr = visualizer.render(cam_bgr, cam_result)
            cam_annotated_pil = bgr_to_pil(cam_annotated_bgr)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Detected Faces", cam_result.total_faces)
            with c2:
                st.metric("Latency", f"{cam_result.inference_time_ms} ms")

            st.image(cam_annotated_pil, caption="Detection Result", use_container_width=True)

            # Crop Gallery for camera
            if cam_result.faces:
                st.markdown("#### Extracted Face Avatars")
                cam_crops = crop_faces(cam_bgr, cam_result.faces)
                crop_cols = st.columns(min(5, max(1, len(cam_crops))))
                for idx, item in enumerate(cam_crops):
                    with crop_cols[idx % len(crop_cols)]:
                        st.image(bgr_to_pil(item["image"]), use_container_width=True)
                        st.caption(f"Face #{item['index']}")

    # ================= TAB 3: VIDEO FILE =================
    with tab_video:
        st.subheader("🎥 Video File Face Detection")
        st.caption("Upload a video clip (MP4, AVI) to process faces across all frames.")

        video_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
        if video_file is not None:
            temp_in_path = "temp_input_video.mp4"
            temp_out_path = "temp_output_annotated.mp4"

            with open(temp_in_path, "wb") as f:
                f.write(video_file.read())

            st.video(temp_in_path)

            if st.button("🚀 Process Video", type="primary"):
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(curr, total):
                    pct = min(1.0, curr / max(1, total))
                    progress_bar.progress(pct)
                    status_text.text(f"Processing frame {curr} of {total} ({int(pct * 100)}%)...")

                with st.spinner("Analyzing and annotating video frames..."):
                    video_stats = process_video_file(
                        input_video_path=temp_in_path,
                        output_video_path=temp_out_path,
                        detector=detector,
                        visualizer=visualizer,
                        progress_callback=update_progress,
                    )

                st.success("✅ Video processing complete!")
                st.write(
                    f"Processed **{video_stats['total_frames']}** frames at **{video_stats['fps']}** FPS with **{video_stats['total_face_instances']}** face sightings."
                )

                if os.path.exists(temp_out_path):
                    with open(temp_out_path, "rb") as vf:
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=vf.read(),
                            file_name="annotated_faces.mp4",
                            mime="video/mp4",
                        )


if __name__ == "__main__":
    main()

