import cv2
import numpy as np
from pathlib import Path

_DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2] / "models" / "sface" / "face_recognition_sface_2021dec.onnx"
)

# Cosine similarity threshold for scene-cut detection.
# OpenCV default (0.363) is too permissive for hard identity cuts in news video.
# Calibrated on genuine val set: same-person min ~0.67, different-person max ~0.43.
# Threshold 0.6 cleanly separates both distributions.
SAME_PERSON_THRESHOLD = 0.6


class FaceReID:
    """
    Scene-cut detector via OpenCV SFace (cv2.FaceRecognizerSF).
    No extra pip installs — ships with opencv-contrib-python.

    Model file (~37 MB): python scripts/download_sface_model.py
    """

    def __init__(self, model_path=None, threshold=SAME_PERSON_THRESHOLD):
        path = Path(model_path) if model_path else _DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"SFace model not found: {path}\n"
                "  → Run:  python scripts/download_sface_model.py"
            )
        self.recognizer = cv2.FaceRecognizerSF.create(str(path), "")
        self.threshold = threshold

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """128-dim feature vector for a face crop (any resolution, BGR)."""
        resized = cv2.resize(face_bgr, (112, 112))
        return self.recognizer.feature(resized)

    def is_same_person(self, feat1: np.ndarray, feat2: np.ndarray) -> bool:
        score = self.recognizer.match(feat1, feat2, cv2.FACE_RECOGNIZER_SF_FR_COSINE)
        return bool(score >= self.threshold)
