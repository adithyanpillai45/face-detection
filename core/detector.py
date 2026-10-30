"""
Unified Face Detection Module supporting both Deep Learning (YuNet ONNX)
and Classical Haar Cascade Classifiers.
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass
class FaceBox:
    """Represents a single detected face."""
    x: int
    y: int
    w: int
    h: int
    score: float = 1.0
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None
    eyes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    smiles: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def box(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def center(self) -> Tuple[int, int]:
        return int(self.x + self.w / 2), int(self.y + self.h / 2)


@dataclass
class DetectionResult:
    """Container for the output of a face detection run."""
    faces: List[FaceBox]
    total_faces: int
    inference_time_ms: float
    image_shape: Tuple[int, int, int]
    model_name: str


class FaceDetector:
    """
    Unified face detection engine.
    
    Supports:
      - 'yunet': Ultra-fast DNN ONNX model with facial landmarks.
      - 'haar': OpenCV Haar Feature-based Cascade Classifier with eyes and smile detection.
    """

    def __init__(
        self,
        model_type: str = "yunet",
        yunet_model_path: Optional[str] = None,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        haar_scale_factor: float = 1.1,
        haar_min_neighbors: int = 5,
        haar_detect_features: bool = True,
    ):
        self.model_type = model_type.lower()
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.haar_scale_factor = haar_scale_factor
        self.haar_min_neighbors = haar_min_neighbors
        self.haar_detect_features = haar_detect_features

        # Resolve model path for YuNet
        if yunet_model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.yunet_model_path = os.path.join(base_dir, "models", "face_detection_yunet.onnx")
        else:
            self.yunet_model_path = yunet_model_path

        self._yunet = None
        self._haar_face = None
        self._haar_eye = None
        self._haar_smile = None

        self._init_models()

    def _init_models(self):
        """Initialize the selected detector model."""
        if self.model_type == "yunet":
            if not os.path.exists(self.yunet_model_path):
                # Fallback to haar if yunet file is missing
                print(f"[FaceDetector] Warning: YuNet model not found at {self.yunet_model_path}. Falling back to Haar.")
                self.model_type = "haar"
                self._init_haar()
            else:
                try:
                    # Initialize YuNet with a dummy size; size is set per frame
                    self._yunet = cv2.FaceDetectorYN.create(
                        model=self.yunet_model_path,
                        config="",
                        input_size=(320, 320),
                        score_threshold=self.conf_threshold,
                        nms_threshold=self.nms_threshold,
                        top_k=5000,
                    )
                except Exception as e:
                    print(f"[FaceDetector] Error loading YuNet: {e}. Falling back to Haar.")
                    self.model_type = "haar"
                    self._init_haar()
        elif self.model_type == "haar":
            self._init_haar()
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose 'yunet' or 'haar'.")

    def _init_haar(self):
        """Initialize Haar cascade classifiers."""
        face_cascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        eye_cascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
        smile_cascade_file = os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml")

        self._haar_face = cv2.CascadeClassifier(face_cascade_file)
        if self.haar_detect_features:
            self._haar_eye = cv2.CascadeClassifier(eye_cascade_file)
            self._haar_smile = cv2.CascadeClassifier(smile_cascade_file)

    def set_confidence(self, conf: float):
        """Update confidence threshold."""
        self.conf_threshold = conf
        if self._yunet is not None:
            self._yunet.setScoreThreshold(conf)

    def set_nms(self, nms: float):
        """Update NMS threshold for YuNet."""
        self.nms_threshold = nms
        if self._yunet is not None:
            self._yunet.setNMSThreshold(nms)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect faces in a BGR or RGB image.
        
        Args:
            image: numpy array (H, W, C) in BGR or RGB format.
            
        Returns:
            DetectionResult containing detected faces and metadata.
        """
        if image is None or image.size == 0:
            return DetectionResult(
                faces=[],
                total_faces=0,
                inference_time_ms=0.0,
                image_shape=(0, 0, 0),
                model_name=self.model_type,
            )

        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1

        t0 = time.perf_counter()

        if self.model_type == "yunet" and self._yunet is not None:
            faces = self._detect_yunet(image, w, h)
        else:
            faces = self._detect_haar(image)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return DetectionResult(
            faces=faces,
            total_faces=len(faces),
            inference_time_ms=round(elapsed_ms, 2),
            image_shape=(h, w, c),
            model_name="YuNet (Deep Learning)" if self.model_type == "yunet" else "Haar Cascade",
        )

    def _detect_yunet(self, image: np.ndarray, w: int, h: int) -> List[FaceBox]:
        """Perform detection using YuNet ONNX model."""
        self._yunet.setInputSize((w, h))
        _, detected = self._yunet.detect(image)

        if detected is None:
            return []

        face_boxes: List[FaceBox] = []
        for face in detected:
            # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
            x, y, fw, fh = map(int, face[0:4])
            score = float(face[14])

            # Clamp coordinates to image boundaries
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)

            landmarks = {
                "right_eye": (int(face[4]), int(face[5])),
                "left_eye": (int(face[6]), int(face[7])),
                "nose_tip": (int(face[8]), int(face[9])),
                "mouth_right": (int(face[10]), int(face[11])),
                "mouth_left": (int(face[12]), int(face[13])),
            }

            face_boxes.append(
                FaceBox(
                    x=x,
                    y=y,
                    w=fw,
                    h=fh,
                    score=score,
                    landmarks=landmarks,
                )
            )

        return face_boxes

    def _detect_haar(self, image: np.ndarray) -> List[FaceBox]:
        """Perform detection using classical Haar cascades."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Equalize histogram for improved contrast
        gray_eq = cv2.equalizeHist(gray)

        raw_faces = self._haar_face.detectMultiScale(
            gray_eq,
            scaleFactor=self.haar_scale_factor,
            minNeighbors=self.haar_min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        face_boxes: List[FaceBox] = []
        if len(raw_faces) == 0:
            return face_boxes

        for (x, y, w, h) in raw_faces:
            eyes_list = []
            smiles_list = []

            if self.haar_detect_features:
                roi_gray = gray_eq[y : y + h, x : x + w]

                # Detect eyes in upper 60% of face
                eye_roi = roi_gray[: int(h * 0.6), :]
                raw_eyes = self._haar_eye.detectMultiScale(eye_roi, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
                for ex, ey, ew, eh in raw_eyes:
                    eyes_list.append((x + ex, y + ey, ew, eh))

                # Detect smile in lower 50% of face
                smile_roi = roi_gray[int(h * 0.5) :, :]
                raw_smiles = self._haar_smile.detectMultiScale(
                    smile_roi, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25)
                )
                for sx, sy, sw, sh in raw_smiles:
                    smiles_list.append((x + sx, y + int(h * 0.5) + sy, sw, sh))

            face_boxes.append(
                FaceBox(
                    x=int(x),
                    y=int(y),
                    w=int(w),
                    h=int(h),
                    score=1.0,
                    eyes=eyes_list,
                    smiles=smiles_list,
                )
            )

        return face_boxes

