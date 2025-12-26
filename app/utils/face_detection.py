"""Face detection integration using the UniFace library.

This module wraps UniFace's RetinaFace and SCRFD detectors so that the rest of the
Smart-NVR pipeline can opt-in to run lightweight face detection without
duplicating integration logic across camera processors.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DetectorType(Enum):
    """Available face detector types."""
    RETINAFACE = "retinaface"
    SCRFD = "scrfd"


@dataclass
class FaceDetectionConfig:
    """Simple value object describing how the detector should behave."""

    enabled: bool = False
    confidence: float = 0.5
    model_name: Optional[str] = None
    provider: str = "auto"
    max_faces: int = 10
    detector_type: str = "retinaface"
    # Quality filtering settings
    min_face_size: int = 32
    blur_detection: bool = False
    bbox_padding: float = 0.1  # 10% padding for better recognition crops

    def config_hash(self) -> str:
        """Generate a hash for caching detector instances."""
        key = f"{self.enabled}:{self.confidence}:{self.model_name}:{self.provider}:{self.detector_type}"
        return hashlib.md5(key.encode()).hexdigest()[:8]


# Global detector cache to avoid reloading models
_DETECTOR_CACHE: Dict[str, "FaceDetectionEngine"] = {}
_DETECTOR_CACHE_LOCK = threading.Lock()


class FaceDetectionEngine:
    """Wrapper around UniFace RetinaFace/SCRFD detector with graceful degradation."""

    def __init__(self, config: FaceDetectionConfig) -> None:
        self.config = config
        self._detector = None
        self._landmarks_supported = False
        self._provider = None
        self._init_error: Optional[Exception] = None
        self._detector_type = DetectorType.RETINAFACE

        if not config.enabled:
            logger.debug("Face detection disabled by configuration; skipping setup")
            return

        try:
            from uniface import RetinaFace
            from uniface.constants import RetinaFaceWeights
            from uniface.onnx_utils import get_available_providers
        except ImportError as exc:
            self._init_error = exc
            logger.warning("UniFace is not installed; face detection will remain disabled: %s", exc)
            return

        # Try to import SCRFD if requested
        scrfd_cls = None
        scrfd_weights = None
        if config.detector_type.lower() == "scrfd":
            try:
                from uniface import SCRFD
                from uniface.constants import SCRFDWeights
                scrfd_cls = SCRFD
                scrfd_weights = SCRFDWeights
                self._detector_type = DetectorType.SCRFD
            except ImportError:
                logger.warning("SCRFD not available in UniFace version; falling back to RetinaFace")

        # Determine which detector to use
        if self._detector_type == DetectorType.SCRFD and scrfd_cls and scrfd_weights:
            model_enum = self._resolve_scrfd_model(config.model_name, scrfd_weights)
            detector_cls = scrfd_cls
        else:
            model_enum = self._resolve_model_enum(config.model_name, RetinaFaceWeights)
            detector_cls = RetinaFace

        provider_key = (config.provider or "auto").lower().strip()
        provider_list = self._resolve_providers(provider_key)

        if provider_list:
            try:
                available = {p.lower(): p for p in get_available_providers()}
            except Exception as exc:  # pragma: no cover - defensive fallback
                available = {}
                logger.debug("Failed to query ONNX providers, continuing with best-effort: %s", exc)

            resolved_providers: List[str] = []
            for provider in provider_list:
                provider_lc = provider.lower()
                if provider_lc in available:
                    resolved_providers.append(available[provider_lc])
                else:
                    resolved_providers.append(provider)
            kwargs: Dict[str, Any] = {
                "conf_thresh": max(0.0, min(1.0, config.confidence)),
                "model_name": model_enum,
                "providers": resolved_providers,
            }
        else:
            kwargs = {
                "conf_thresh": max(0.0, min(1.0, config.confidence)),
                "model_name": model_enum,
            }

        try:
            detector = detector_cls(**kwargs)
            self._detector = detector
            self._landmarks_supported = getattr(detector, "supports_landmarks", True)
            self._provider = kwargs.get("providers")
            provider_repr = self._provider or "auto"
            logger.info("UniFace %s initialised (model=%s, provider=%s)", 
                       self._detector_type.value, model_enum, provider_repr)
        except Exception as exc:  # pragma: no cover - runtime safety
            self._init_error = exc
            logger.error("Failed to initialise UniFace %s: %s", self._detector_type.value, exc, exc_info=True)

    @property
    def ready(self) -> bool:
        """Return True when a detector instance is available."""

        return self._detector is not None

    @property
    def init_error(self) -> Optional[Exception]:
        """Expose the initialisation error (if any) for downstream logging."""

        return self._init_error

    @property
    def detector_type(self) -> DetectorType:
        """Return the active detector type."""
        return self._detector_type

    def detect(self, frame, frame_height: int = 0, frame_width: int = 0) -> List[Dict[str, Any]]:
        """Run face detection on a single frame.

        Returns a list of dictionaries following the Smart-NVR detection schema.
        """

        if not self.ready:
            return []

        if frame_height == 0 or frame_width == 0:
            frame_height, frame_width = frame.shape[:2]

        try:
            faces: List[Dict[str, Any]] = self._detector.detect(
                frame,
                max_num=self.config.max_faces if self.config.max_faces > 0 else 0,
            )
        except Exception as exc:  # pragma: no cover - runtime protection
            logger.error("UniFace detection failure: %s", exc, exc_info=True)
            return []

        detections: List[Dict[str, Any]] = []
        for face in faces:
            bbox = face.get("bbox") or []
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            confidence = float(face.get("confidence", 0.0))
            if confidence < self.config.confidence:
                continue

            width = max(0.0, float(x2) - float(x1))
            height = max(0.0, float(y2) - float(y1))

            # Quality filter: minimum face size
            if width < self.config.min_face_size or height < self.config.min_face_size:
                logger.debug("Skipping small face: %.0fx%.0f < %d", width, height, self.config.min_face_size)
                continue

            # Apply blur detection if enabled
            if self.config.blur_detection:
                is_blurry = self._check_blur(frame, int(x1), int(y1), int(x2), int(y2))
                if is_blurry:
                    logger.debug("Skipping blurry face at (%.0f, %.0f)", x1, y1)
                    continue

            # Apply bounding box padding for better recognition crops
            if self.config.bbox_padding > 0:
                x1, y1, x2, y2 = self._apply_padding(
                    x1, y1, x2, y2, 
                    frame_width, frame_height, 
                    self.config.bbox_padding
                )
                width = max(0.0, float(x2) - float(x1))
                height = max(0.0, float(y2) - float(y1))

            detection: Dict[str, Any] = {
                "class_name": "face",
                "confidence": confidence,
                "bbox_x": float(x1),
                "bbox_y": float(y1),
                "bbox_width": width,
                "bbox_height": height,
                "detector_type": self._detector_type.value,
            }

            if self._landmarks_supported and face.get("landmarks"):
                detection["landmarks"] = face["landmarks"]

            detections.append(detection)

        return detections

    def _check_blur(self, frame, x1: int, y1: int, x2: int, y2: int, threshold: float = 100.0) -> bool:
        """Check if a face region is blurry using Laplacian variance."""
        try:
            import cv2
            
            # Clamp coordinates
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return True
                
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian_var < threshold
        except Exception:
            return False  # On error, don't filter

    @staticmethod
    def _apply_padding(x1: float, y1: float, x2: float, y2: float, 
                       frame_w: int, frame_h: int, padding: float) -> Tuple[float, float, float, float]:
        """Apply percentage padding to bounding box while keeping within frame bounds."""
        width = x2 - x1
        height = y2 - y1
        pad_w = width * padding
        pad_h = height * padding
        
        new_x1 = max(0.0, x1 - pad_w)
        new_y1 = max(0.0, y1 - pad_h)
        new_x2 = min(float(frame_w), x2 + pad_w)
        new_y2 = min(float(frame_h), y2 + pad_h)
        
        return new_x1, new_y1, new_x2, new_y2

    @staticmethod
    def _resolve_model_enum(model_name: Optional[str], enum_cls) -> Any:
        if not model_name:
            return enum_cls.MNET_V2 if hasattr(enum_cls, "MNET_V2") else None

        candidate = model_name.upper().strip().replace("-", "_")
        if not candidate:
            return enum_cls.MNET_V2 if hasattr(enum_cls, "MNET_V2") else None

        if hasattr(enum_cls, candidate):
            return getattr(enum_cls, candidate)

        logger.warning("Unknown RetinaFace model '%s', falling back to default", model_name)
        return enum_cls.MNET_V2 if hasattr(enum_cls, "MNET_V2") else None

    @staticmethod
    def _resolve_scrfd_model(model_name: Optional[str], enum_cls) -> Any:
        """Resolve SCRFD model name to enum value."""
        if not model_name:
            # Default to SCRFD_2.5G_KPS for good balance
            return getattr(enum_cls, "SCRFD_2_5G_KPS", None) or getattr(enum_cls, "SCRFD_10G_KPS", None)

        candidate = model_name.upper().strip().replace("-", "_").replace(".", "_")
        if hasattr(enum_cls, candidate):
            return getattr(enum_cls, candidate)

        # Try common variations
        for attr in ["SCRFD_2_5G_KPS", "SCRFD_10G_KPS", "SCRFD_500M_KPS"]:
            if hasattr(enum_cls, attr):
                logger.warning("Unknown SCRFD model '%s', falling back to %s", model_name, attr)
                return getattr(enum_cls, attr)
        
        return None

    @staticmethod
    def _resolve_providers(provider_key: str) -> Optional[List[str]]:
        if not provider_key or provider_key == "auto":
            return None

        provider_map = {
            "cpu": ["CPUExecutionProvider"],
            "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
        }

        provider_list = provider_map.get(provider_key)
        if not provider_list:
            logger.warning("Unsupported UniFace provider '%s', defaulting to auto", provider_key)
            return None

        return provider_list


def build_face_detection_engine(settings: Dict[str, Any]) -> FaceDetectionEngine:
    """Factory helper to keep CameraProcessor clean.
    
    Uses caching to avoid reloading models when settings haven't changed.
    """

    config = FaceDetectionConfig(
        enabled=settings.get("enable_face_detection", False),
        confidence=float(settings.get("face_detection_confidence", 0.5) or 0.5),
        model_name=settings.get("face_detection_model"),
        provider=settings.get("face_detection_provider", "auto"),
        max_faces=int(settings.get("face_detection_max_faces", 10) or 10),
        detector_type=settings.get("face_detector_type", "retinaface"),
        min_face_size=int(settings.get("face_min_size", 32) or 32),
        blur_detection=bool(settings.get("face_blur_detection", False)),
        bbox_padding=float(settings.get("face_bbox_padding", 0.1) or 0.1),
    )
    
    # Check cache for existing engine with same config
    cache_key = config.config_hash()
    with _DETECTOR_CACHE_LOCK:
        if cache_key in _DETECTOR_CACHE:
            cached = _DETECTOR_CACHE[cache_key]
            if cached.ready:
                logger.debug("Reusing cached face detection engine (key=%s)", cache_key)
                return cached
    
    # Create new engine and cache it
    engine = FaceDetectionEngine(config)
    if engine.ready:
        with _DETECTOR_CACHE_LOCK:
            _DETECTOR_CACHE[cache_key] = engine
    
    return engine


def clear_detector_cache() -> int:
    """Clear all cached detector instances. Returns count of cleared entries."""
    with _DETECTOR_CACHE_LOCK:
        count = len(_DETECTOR_CACHE)
        _DETECTOR_CACHE.clear()
        return count
