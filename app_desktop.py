"""
VisionAI - Native Desktop GUI Face Detection Application
Built with Tkinter, OpenCV, and Pillow. Works natively on Windows without extra dependencies.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from core.detector import FaceDetector, DetectionResult
from core.visualizer import FaceVisualizer, THEME_COLORS
from core.utils import load_image, save_image, crop_faces


class FaceDetectionDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VisionAI • Face Detection Desktop Studio")
        self.root.geometry("1150x780")
        self.root.minsize(950, 650)
        self.root.configure(bg="#12141a")

        # Application state
        self.current_image_bgr: np.ndarray = None
        self.annotated_image_bgr: np.ndarray = None
        self.last_result: DetectionResult = None
        self.cap: cv2.VideoCapture = None
        self.webcam_active = False

        # Detector and Visualizer state
        self.model_var = tk.StringVar(value="yunet")
        self.conf_var = tk.DoubleVar(value=0.6)
        self.theme_var = tk.StringVar(value="Cyber Neon")
        self.box_style_var = tk.StringVar(value="corner_hud")
        self.line_thick_var = tk.IntVar(value=2)
        self.landmarks_var = tk.BooleanVar(value=True)
        self.scores_var = tk.BooleanVar(value=True)
        self.hud_var = tk.BooleanVar(value=True)
        self.dim_bg_var = tk.BooleanVar(value=False)
        self.privacy_var = tk.StringVar(value="none")
        self.blur_strength_var = tk.IntVar(value=35)

        self.detector = FaceDetector(model_type=self.model_var.get(), conf_threshold=self.conf_var.get())
        self.visualizer = FaceVisualizer(
            theme=self.theme_var.get(),
            box_style=self.box_style_var.get(),
            line_thickness=self.line_thick_var.get(),
            draw_landmarks=self.landmarks_var.get(),
            draw_scores=self.scores_var.get(),
            draw_hud_banner=self.hud_var.get(),
            privacy_mode=self.privacy_var.get(),
            blur_strength=self.blur_strength_var.get(),
            dim_background=self.dim_bg_var.get(),
        )

        self._setup_styles()
        self._build_ui()
        self._load_default_sample()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#12141a", foreground="#e2e8f0", font=("Segoe UI", 9))
        self.style.configure("TLabel", background="#12141a", foreground="#cbd5e1")
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#38bdf8")
        self.style.configure("Card.TFrame", background="#1a1d24", relief="flat")
        self.style.configure("TButton", background="#2563eb", foreground="#ffffff", font=("Segoe UI", 9, "bold"), borderwidth=0)
        self.style.map("TButton", background=[("active", "#1d4ed8")])
        self.style.configure("Secondary.TButton", background="#334155", foreground="#ffffff")
        self.style.map("Secondary.TButton", background=[("active", "#475569")])
        self.style.configure("Action.TButton", background="#059669", foreground="#ffffff")
        self.style.map("Action.TButton", background=[("active", "#047857")])

    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root, bg="#1a1d24", height=50)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

        title_lbl = tk.Label(
            toolbar,
            text="👁️ VisionAI Studio",
            font=("Segoe UI", 14, "bold"),
            bg="#1a1d24",
            fg="#38bdf8",
        )
        title_lbl.pack(side=tk.LEFT, padx=15, pady=8)

        btn_open = tk.Button(
            toolbar, text="📂 Open Image", bg="#2563eb", fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=12, pady=5, command=self.open_image_dialog
        )
        btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_webcam = tk.Button(
            toolbar, text="📹 Start Webcam", bg="#059669", fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=12, pady=5, command=self.toggle_webcam
        )
        self.btn_webcam.pack(side=tk.LEFT, padx=5)

        btn_sample = tk.Button(
            toolbar, text="🖼️ Sample Image", bg="#475569", fg="white", font=("Segoe UI", 9),
            relief="flat", padx=10, pady=5, command=self._load_default_sample
        )
        btn_sample.pack(side=tk.LEFT, padx=5)

        btn_save = tk.Button(
            toolbar, text="💾 Save Output", bg="#0284c7", fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=12, pady=5, command=self.save_output_image
        )
        btn_save.pack(side=tk.RIGHT, padx=15)

        # Main Layout: Sidebar & Preview
        content_frame = tk.Frame(self.root, bg="#12141a")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left Sidebar (Controls)
        sidebar = tk.Frame(content_frame, bg="#1a1d24", width=310)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        # Model Selection Group
        lbl_sec1 = tk.Label(sidebar, text="MODEL CONFIGURATION", font=("Segoe UI", 10, "bold"), bg="#1a1d24", fg="#38bdf8")
        lbl_sec1.pack(anchor="w", padx=14, pady=(14, 6))

        rb1 = tk.Radiobutton(
            sidebar, text="YuNet Deep Learning (Fast & Acc)", variable=self.model_var, value="yunet",
            bg="#1a1d24", fg="#f8fafc", selectcolor="#0f172a", font=("Segoe UI", 9),
            command=self.on_settings_change
        )
        rb1.pack(anchor="w", padx=16, pady=2)

        rb2 = tk.Radiobutton(
            sidebar, text="Haar Cascade (Classical CV)", variable=self.model_var, value="haar",
            bg="#1a1d24", fg="#f8fafc", selectcolor="#0f172a", font=("Segoe UI", 9),
            command=self.on_settings_change
        )
        rb2.pack(anchor="w", padx=16, pady=2)

        # Threshold slider
        tk.Label(sidebar, text="Confidence Threshold:", bg="#1a1d24", fg="#94a3b8").pack(anchor="w", padx=16, pady=(10, 2))
        s_conf = tk.Scale(
            sidebar, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.conf_var, bg="#1a1d24", fg="#38bdf8", highlightthickness=0,
            troughcolor="#0f172a", command=lambda v: self.on_settings_change()
        )
        s_conf.pack(fill=tk.X, padx=16)

        # Styling Group
        tk.Label(sidebar, text="VISUAL STYLING", font=("Segoe UI", 10, "bold"), bg="#1a1d24", fg="#38bdf8").pack(anchor="w", padx=14, pady=(16, 6))

        tk.Label(sidebar, text="Color Theme:", bg="#1a1d24", fg="#94a3b8").pack(anchor="w", padx=16, pady=(2, 2))
        theme_cb = ttk.Combobox(sidebar, textvariable=self.theme_var, values=list(THEME_COLORS.keys()), state="readonly")
        theme_cb.pack(fill=tk.X, padx=16)
        theme_cb.bind("<<ComboboxSelected>>", lambda e: self.on_settings_change())

        tk.Label(sidebar, text="Bounding Box Style:", bg="#1a1d24", fg="#94a3b8").pack(anchor="w", padx=16, pady=(8, 2))
        style_cb = ttk.Combobox(sidebar, textvariable=self.box_style_var, values=["corner_hud", "rounded", "rectangle"], state="readonly")
        style_cb.pack(fill=tk.X, padx=16)
        style_cb.bind("<<ComboboxSelected>>", lambda e: self.on_settings_change())

        # Checkboxes
        chk_land = tk.Checkbutton(
            sidebar, text="Show Facial Landmarks", variable=self.landmarks_var,
            bg="#1a1d24", fg="#f8fafc", selectcolor="#0f172a", font=("Segoe UI", 9),
            command=self.on_settings_change
        )
        chk_land.pack(anchor="w", padx=16, pady=(8, 2))

        chk_score = tk.Checkbutton(
            sidebar, text="Show Confidence Scores", variable=self.scores_var,
            bg="#1a1d24", fg="#f8fafc", selectcolor="#0f172a", font=("Segoe UI", 9),
            command=self.on_settings_change
        )
        chk_score.pack(anchor="w", padx=16, pady=2)

        chk_hud = tk.Checkbutton(
            sidebar, text="Show Top Status Bar", variable=self.hud_var,
            bg="#1a1d24", fg="#f8fafc", selectcolor="#0f172a", font=("Segoe UI", 9),
            command=self.on_settings_change
        )
        chk_hud.pack(anchor="w", padx=16, pady=2)

        # Privacy Filter
        tk.Label(sidebar, text="PRIVACY ANONYMIZATION", font=("Segoe UI", 10, "bold"), bg="#1a1d24", fg="#38bdf8").pack(anchor="w", padx=14, pady=(16, 6))

        for priv_opt in [("No Privacy Filter", "none"), ("Blur Faces", "blur"), ("Pixelate / Mosaic", "mosaic")]:
            tk.Radiobutton(
                sidebar, text=priv_opt[0], variable=self.privacy_var, value=priv_opt[1],
                bg="#1a1d24", fg="#f8fafc", selectcolor="#0f172a", font=("Segoe UI", 9),
                command=self.on_settings_change
            ).pack(anchor="w", padx=16, pady=2)

        # Stats Card
        stats_frame = tk.Frame(sidebar, bg="#0f172a", bd=1, relief="solid")
        stats_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=14)

        self.lbl_face_count = tk.Label(stats_frame, text="Faces: 0", font=("Segoe UI", 13, "bold"), bg="#0f172a", fg="#00f2fe")
        self.lbl_face_count.pack(pady=(8, 2))

        self.lbl_latency = tk.Label(stats_frame, text="Latency: 0 ms", font=("Segoe UI", 9), bg="#0f172a", fg="#94a3b8")
        self.lbl_latency.pack(pady=(0, 8))

        # Right Viewport (Image Display Canvas)
        view_frame = tk.Frame(content_frame, bg="#090a0f")
        view_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(view_frame, bg="#090a0f", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Status Bar
        self.status_bar = tk.Label(self.root, text="Ready. Open an image or start webcam.", bg="#0f172a", fg="#94a3b8", anchor="w", padx=12, pady=4)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _load_default_sample(self):
        """Load default portrait sample from samples directory."""
        sample_path = os.path.join(os.path.dirname(__file__), "samples", "sample_portrait.jpg")
        if os.path.exists(sample_path):
            self.load_image_from_path(sample_path)
        else:
            # Create a blank fallback canvas
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No Image Loaded", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
            self.current_image_bgr = blank
            self.process_and_render()

    def open_image_dialog(self):
        """Open file dialog for user to choose an image."""
        if self.webcam_active:
            self.stop_webcam()

        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp"), ("All Files", "*.*")],
        )
        if file_path:
            self.load_image_from_path(file_path)

    def load_image_from_path(self, path: str):
        try:
            self.current_image_bgr = load_image(path)
            self.status_bar.config(text=f"Loaded: {os.path.basename(path)} ({self.current_image_bgr.shape[1]}x{self.current_image_bgr.shape[0]})")
            self.process_and_render()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def on_settings_change(self):
        """Callback when any control setting changes."""
        self.detector = FaceDetector(
            model_type=self.model_var.get(),
            conf_threshold=self.conf_var.get(),
        )
        self.visualizer = FaceVisualizer(
            theme=self.theme_var.get(),
            box_style=self.box_style_var.get(),
            line_thickness=self.line_thick_var.get(),
            draw_landmarks=self.landmarks_var.get(),
            draw_scores=self.scores_var.get(),
            draw_hud_banner=self.hud_var.get(),
            privacy_mode=self.privacy_var.get(),
            blur_strength=self.blur_strength_var.get(),
            dim_background=self.dim_bg_var.get(),
        )
        if not self.webcam_active and self.current_image_bgr is not None:
            self.process_and_render()

    def process_and_render(self):
        """Run detection on current image and render to canvas."""
        if self.current_image_bgr is None:
            return

        self.last_result = self.detector.detect(self.current_image_bgr)
        self.annotated_image_bgr = self.visualizer.render(self.current_image_bgr, self.last_result)

        # Update stats
        self.lbl_face_count.config(text=f"Faces: {self.last_result.total_faces}")
        self.lbl_latency.config(text=f"Latency: {self.last_result.inference_time_ms} ms")

        self.draw_image_on_canvas(self.annotated_image_bgr)

    def draw_image_on_canvas(self, bgr_img: np.ndarray):
        """Fit image inside canvas preserving aspect ratio."""
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        h, w = bgr_img.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img_pil = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(image=img_pil)

        self.canvas.delete("all")
        # Center image
        x_pos = (cw - nw) // 2
        y_pos = (ch - nh) // 2
        self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_image)

    def on_canvas_resize(self, event):
        """Handle canvas resizing."""
        if self.annotated_image_bgr is not None and not self.webcam_active:
            self.draw_image_on_canvas(self.annotated_image_bgr)

    def toggle_webcam(self):
        """Toggle live webcam streaming."""
        if self.webcam_active:
            self.stop_webcam()
        else:
            self.start_webcam()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showwarning("Webcam Warning", "Could not open webcam device #0. Please ensure a camera is connected.")
            return

        self.webcam_active = True
        self.btn_webcam.config(text="⏹️ Stop Webcam", bg="#dc2626")
        self.status_bar.config(text="Webcam active. Detecting faces in real time...")
        self._webcam_loop()

    def stop_webcam(self):
        self.webcam_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_webcam.config(text="📹 Start Webcam", bg="#059669")
        self.status_bar.config(text="Webcam stopped.")

    def _webcam_loop(self):
        if not self.webcam_active or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            # Mirror frame horizontally for natural webcam feel
            frame = cv2.flip(frame, 1)
            self.current_image_bgr = frame
            self.last_result = self.detector.detect(frame)
            self.annotated_image_bgr = self.visualizer.render(frame, self.last_result)

            self.lbl_face_count.config(text=f"Faces: {self.last_result.total_faces}")
            self.lbl_latency.config(text=f"Latency: {self.last_result.inference_time_ms} ms")
            self.draw_image_on_canvas(self.annotated_image_bgr)

        self.root.after(20, self._webcam_loop)

    def save_output_image(self):
        """Save current annotated output to disk."""
        if self.annotated_image_bgr is None:
            messagebox.showinfo("Info", "No processed image to save.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile="face_detection_result.jpg",
        )
        if file_path:
            save_image(self.annotated_image_bgr, file_path)
            self.status_bar.config(text=f"Saved result to: {file_path}")
            messagebox.showinfo("Success", f"Annotated image saved successfully!\n{file_path}")

    def on_closing(self):
        self.stop_webcam()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FaceDetectionDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

