"""
High-quality visualization engine for face detection, HUD overlays,
facial landmarks, and privacy filters (blur/pixelate).
"""

from typing import Tuple, Optional
import cv2
import numpy as np
try:
    from .detector import DetectionResult, FaceBox
except (ImportError, ValueError):
    from detector import DetectionResult, FaceBox


THEME_COLORS = {
    "Cyber Neon": (57, 255, 20),      # Bright Neon Green (BGR: 20, 255, 57)
    "Electric Cyan": (255, 255, 0),    # Cyan (BGR: 255, 255, 0)
    "Sunset Orange": (0, 140, 255),    # Orange (BGR: 0, 140, 255)
    "Ultra Violet": (255, 0, 180),     # Purple/Magenta (BGR: 180, 0, 255)
    "Alert Red": (40, 40, 255),        # Red (BGR: 40, 40, 255)
    "Clean White": (255, 255, 255),    # White (BGR: 255, 255, 255)
}


class FaceVisualizer:
    """
    Renders bounding boxes, landmarks, HUD overlays, and anonymization filters.
    """

    def __init__(
        self,
        theme: str = "Cyber Neon",
        box_style: str = "corner_hud",  # 'corner_hud', 'rectangle', 'rounded'
        line_thickness: int = 2,
        draw_landmarks: bool = True,
        draw_scores: bool = True,
        draw_hud_banner: bool = True,
        privacy_mode: str = "none",  # 'none', 'blur', 'mosaic'
        blur_strength: int = 35,
        mosaic_block_size: int = 15,
        dim_background: bool = False,
    ):
        self.theme_name = theme
        self.primary_color = THEME_COLORS.get(theme, (57, 255, 20))
        self.box_style = box_style
        self.line_thickness = line_thickness
        self.draw_landmarks = draw_landmarks
        self.draw_scores = draw_scores
        self.draw_hud_banner = draw_hud_banner
        self.privacy_mode = privacy_mode
        self.blur_strength = max(3, blur_strength | 1)  # must be odd
        self.mosaic_block_size = max(4, mosaic_block_size)
        self.dim_background = dim_background

    def set_theme(self, theme: str):
        """Update active color theme."""
        if theme in THEME_COLORS:
            self.theme_name = theme
            self.primary_color = THEME_COLORS[theme]

    def render(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        """
        Render detection results on an image copy.
        
        Args:
            image: input image (numpy array, BGR)
            result: DetectionResult instance
            
        Returns:
            annotated image (numpy array, BGR)
        """
        canvas = image.copy()
        h, w = canvas.shape[:2]

        # 1. Dim background if enabled
        if self.dim_background and result.faces:
            canvas = self._apply_dim_background(canvas, result.faces)

        # 2. Apply privacy filter (blur or mosaic) if enabled
        if self.privacy_mode in ("blur", "mosaic") and result.faces:
            canvas = self._apply_privacy_filter(canvas, result.faces)

        # 3. Draw face boxes, landmarks, and labels
        for idx, face in enumerate(result.faces, start=1):
            self._draw_face(canvas, face, idx)

        # 4. Draw HUD banner if enabled
        if self.draw_hud_banner:
            self._draw_hud_banner(canvas, result)

        return canvas

    def _draw_face(self, canvas: np.ndarray, face: FaceBox, idx: int):
        """Draw bounding box, HUD corners, score pill, and landmarks for a face."""
        x, y, w, h = face.x, face.y, face.w, face.h
        color = self.primary_color
        t = self.line_thickness

        # Draw bounding box style
        if self.box_style == "corner_hud":
            self._draw_corner_hud(canvas, x, y, w, h, color, t)
        elif self.box_style == "rounded":
            self._draw_rounded_rect(canvas, (x, y), (x + w, y + h), color, t, radius=12)
        else:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, t)

        # Draw label pill
        label = f"Face #{idx}"
        if self.draw_scores and face.score < 1.0:
            label += f" ({int(face.score * 100)}%)"

        self._draw_label_pill(canvas, label, x, y, color)

        # Draw landmarks if available
        if self.draw_landmarks and face.landmarks:
            self._draw_landmarks(canvas, face.landmarks)

        # Draw eye & smile boxes if Haar detected them
        if face.eyes:
            for (ex, ey, ew, eh) in face.eyes:
                cv2.rectangle(canvas, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 1)
        if face.smiles:
            for (sx, sy, sw, sh) in face.smiles:
                cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), (0, 200, 255), 1)

    def _draw_corner_hud(self, canvas: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int], t: int):
        """Draw sleek futuristic HUD corner brackets with slight glow."""
        corner_len = max(12, int(min(w, h) * 0.22))
        
        # Subtle semi-transparent bounding fill
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        # Top-Left corner
        cv2.line(canvas, (x, y), (x + corner_len, y), color, t + 1)
        cv2.line(canvas, (x, y), (x, y + corner_len), color, t + 1)

        # Top-Right corner
        cv2.line(canvas, (x + w, y), (x + w - corner_len, y), color, t + 1)
        cv2.line(canvas, (x + w, y), (x + w, y + corner_len), color, t + 1)

        # Bottom-Left corner
        cv2.line(canvas, (x, y + h), (x + corner_len, y + h), color, t + 1)
        cv2.line(canvas, (x, y + h), (x, y + h - corner_len), color, t + 1)

        # Bottom-Right corner
        cv2.line(canvas, (x + w, y + h), (x + w - corner_len, y + h), color, t + 1)
        cv2.line(canvas, (x + w, y + h), (x + w, y + h - corner_len), color, t + 1)

    def _draw_rounded_rect(self, canvas: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], t: int, radius: int = 10):
        """Draw rectangle with rounded corners."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = min(radius, abs(x2 - x1) // 4, abs(y2 - y1) // 4)

        # Straight segments
        cv2.line(canvas, (x1 + r, y1), (x2 - r, y1), color, t)
        cv2.line(canvas, (x1 + r, y2), (x2 - r, y2), color, t)
        cv2.line(canvas, (x1, y1 + r), (x1, y2 - r), color, t)
        cv2.line(canvas, (x2, y1 + r), (x2, y2 - r), color, t)

        # Corner arcs
        cv2.ellipse(canvas, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, t)
        cv2.ellipse(canvas, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, t)
        cv2.ellipse(canvas, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, t)
        cv2.ellipse(canvas, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t)

    def _draw_label_pill(self, canvas: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int]):
        """Render a readable label badge above the detected face."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        pad_x, pad_y = 6, 4

        pill_y2 = max(y - 5, text_h + pad_y * 2)
        pill_y1 = pill_y2 - text_h - pad_y * 2
        pill_x1 = max(0, x)
        pill_x2 = min(canvas.shape[1], pill_x1 + text_w + pad_x * 2)

        # Background badge (dark with slight color border)
        cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), (20, 20, 20), -1)
        cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), color, 1)

        # Text
        cv2.putText(
            canvas,
            text,
            (pill_x1 + pad_x, pill_y2 - pad_y - 1),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    def _draw_landmarks(self, canvas: np.ndarray, landmarks: dict):
        """Draw facial landmark keypoints."""
        # Feature colors: Eyes (Cyan), Nose (Yellow), Mouth (Orange)
        palette = {
            "right_eye": (255, 255, 0),
            "left_eye": (255, 255, 0),
            "nose_tip": (0, 255, 255),
            "mouth_right": (0, 165, 255),
            "mouth_left": (0, 165, 255),
        }

        # Subtle connecting lines for facial geometry
        re = landmarks.get("right_eye")
        le = landmarks.get("left_eye")
        nt = landmarks.get("nose_tip")
        mr = landmarks.get("mouth_right")
        ml = landmarks.get("mouth_left")

        if re and le:
            cv2.line(canvas, re, le, (200, 200, 200), 1, cv2.LINE_AA)
        if re and nt:
            cv2.line(canvas, re, nt, (180, 180, 180), 1, cv2.LINE_AA)
        if le and nt:
            cv2.line(canvas, le, nt, (180, 180, 180), 1, cv2.LINE_AA)
        if mr and ml:
            cv2.line(canvas, mr, ml, (200, 200, 200), 1, cv2.LINE_AA)
        if nt and mr:
            cv2.line(canvas, nt, mr, (180, 180, 180), 1, cv2.LINE_AA)
        if nt and ml:
            cv2.line(canvas, nt, ml, (180, 180, 180), 1, cv2.LINE_AA)

        # Draw keypoint dots
        for name, pt in landmarks.items():
            pt_color = palette.get(name, (0, 255, 0))
            cv2.circle(canvas, pt, 4, pt_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

    def _apply_privacy_filter(self, canvas: np.ndarray, faces: list) -> np.ndarray:
        """Apply Gaussian blur or pixelation/mosaic to detected face regions."""
        h_img, w_img = canvas.shape[:2]

        for face in faces:
            x, y, w, h = face.x, face.y, face.w, face.h
            if w <= 0 or h <= 0:
                continue

            # Ensure valid bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)

            roi = canvas[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            if self.privacy_mode == "blur":
                k = self.blur_strength | 1  # ensure odd
                blurred = cv2.GaussianBlur(roi, (k, k), 30)
                canvas[y1:y2, x1:x2] = blurred
            elif self.privacy_mode == "mosaic":
                block = max(4, self.mosaic_block_size)
                rh, rw = roi.shape[:2]
                small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_LINEAR)
                mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
                canvas[y1:y2, x1:x2] = mosaic

        return canvas

    def _apply_dim_background(self, canvas: np.ndarray, faces: list) -> np.ndarray:
        """Darken background while highlighting detected faces."""
        h, w = canvas.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for face in faces:
            cv2.rectangle(mask, (face.x, face.y), (face.x + face.w, face.y + face.h), 255, -1)

        dimmed = (canvas * 0.35).astype(np.uint8)
        canvas = np.where(mask[:, :, None] == 255, canvas, dimmed)
        return canvas

    def _draw_hud_banner(self, canvas: np.ndarray, result: DetectionResult):
        """Draw top status bar HUD."""
        w = canvas.shape[1]
        banner_h = 32

        # Semi-transparent dark bar
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.line(canvas, (0, banner_h), (w, banner_h), self.primary_color, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"MODEL: {result.model_name}  |  DETECTED: {result.total_faces} FACE(S)  |  LATENCY: {result.inference_time_ms} ms"
        cv2.putText(canvas, text, (14, 21), font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

