"""Face recognition helper built around facenet-pytorch embeddings."""
from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:  # Optional dependency
    import torch
    from facenet_pytorch import InceptionResnetV1
except ImportError:  # pragma: no cover - environment without facenet
    torch = None
    InceptionResnetV1 = None


_ENGINE_LOCK = threading.Lock()
_SHARED_ENGINE: Optional["FaceRecognitionEngine"] = None


class FaceRecognitionEngine:
    """Singleton-like wrapper for embedding faces and computing similarity."""

    def __init__(self) -> None:
        self._ready = False
        self._init_error: Optional[Exception] = None
        self.device = None
        self.model = None

        if torch is None or InceptionResnetV1 is None:
            self._init_error = ImportError("facenet-pytorch is not installed")
            logger.warning("Face recognition disabled: %s", self._init_error)
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
            self.device = device
            self.model = model
            self._ready = True
            logger.info("Face recognition model loaded on %s", device)
        except Exception as exc:  # pragma: no cover - runtime fallback
            self._init_error = exc
            logger.error("Failed to initialise face recognition model: %s", exc, exc_info=True)

    @property
    def ready(self) -> bool:
        return bool(self._ready and self.model is not None)

    @property
    def init_error(self) -> Optional[Exception]:
        return self._init_error

    def embed(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        if not self.ready:
            return None
        if face_image.size == 0:
            return None

        try:
            import cv2

            resized = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            tensor = (tensor - 0.5) / 0.5
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(tensor).cpu().numpy().astype("float32")[0]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Failed to embed face crop: %s", exc, exc_info=True)
            return None

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        if vec_a is None or vec_b is None:
            return -1.0
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom == 0:
            return -1.0
        return float(np.dot(vec_a, vec_b) / denom)


def get_face_recognition_engine() -> Optional[FaceRecognitionEngine]:
    """Return a shared engine instance if available."""
    global _SHARED_ENGINE
    with _ENGINE_LOCK:
        if _SHARED_ENGINE is None:
            engine = FaceRecognitionEngine()
            if not engine.ready:
                _SHARED_ENGINE = engine
                return engine if engine.ready else None
            _SHARED_ENGINE = engine
        return _SHARED_ENGINE if _SHARED_ENGINE.ready else None