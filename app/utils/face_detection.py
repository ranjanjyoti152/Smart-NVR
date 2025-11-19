"""Face detection integration using the UniFace library.

This module wraps UniFace's RetinaFace detector so that the rest of the
Smart-NVR pipeline can opt-in to run lightweight face detection without
duplicating integration logic across camera processors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FaceDetectionConfig:
    """Simple value object describing how the detector should behave."""

    enabled: bool = False
    confidence: float = 0.5
    model_name: Optional[str] = None
    provider: str = "auto"
    max_faces: int = 10


class FaceDetectionEngine:
    """Wrapper around UniFace RetinaFace detector with graceful degradation."""

    def __init__(self, config: FaceDetectionConfig) -> None:
        self.config = config
        self._detector = None
        self._landmarks_supported = False
        self._provider = None
        self._init_error: Optional[Exception] = None

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

        model_enum = self._resolve_model_enum(config.model_name, RetinaFaceWeights)
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
            detector = RetinaFace(**kwargs)
            self._detector = detector
            self._landmarks_supported = getattr(detector, "supports_landmarks", True)
            self._provider = kwargs.get("providers")
            provider_repr = self._provider or "auto"
            logger.info("UniFace RetinaFace initialised (model=%s, provider=%s)", model_enum, provider_repr)
        except Exception as exc:  # pragma: no cover - runtime safety
            self._init_error = exc
            logger.error("Failed to initialise UniFace RetinaFace: %s", exc, exc_info=True)

    @property
    def ready(self) -> bool:
        """Return True when a detector instance is available."""

        return self._detector is not None

    @property
    def init_error(self) -> Optional[Exception]:
        """Expose the initialisation error (if any) for downstream logging."""

        return self._init_error

    def detect(self, frame) -> List[Dict[str, Any]]:
        """Run face detection on a single frame.

        Returns a list of dictionaries following the Smart-NVR detection schema.
        """

        if not self.ready:
            return []

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

            detection: Dict[str, Any] = {
                "class_name": "face",
                "confidence": confidence,
                "bbox_x": float(x1),
                "bbox_y": float(y1),
                "bbox_width": width,
                "bbox_height": height,
            }

            if self._landmarks_supported and face.get("landmarks"):
                detection["landmarks"] = face["landmarks"]

            detections.append(detection)

        return detections

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
    """Factory helper to keep CameraProcessor clean."""

    config = FaceDetectionConfig(
        enabled=settings.get("enable_face_detection", False),
        confidence=float(settings.get("face_detection_confidence", 0.5) or 0.5),
        model_name=settings.get("face_detection_model"),
        provider=settings.get("face_detection_provider", "auto"),
        max_faces=int(settings.get("face_detection_max_faces", 10) or 10),
    )
    return FaceDetectionEngine(config)
