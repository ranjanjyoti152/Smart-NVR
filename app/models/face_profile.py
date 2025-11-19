"""MongoDB-backed face profile model used for face recognition."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import logging
import os

import numpy as np
from bson import ObjectId

from app import db

logger = logging.getLogger(__name__)


@dataclass
class FaceSample:
    image_path: Optional[str]
    timestamp: datetime
    camera_id: Optional[str] = None
    detection_id: Optional[str] = None


class FaceProfile:
    """Represents a persistent face identity tracked across detections."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._id: Optional[ObjectId] = data.get("_id")
        self.name: Optional[str] = data.get("name")
        self.embedding: List[float] = data.get("embedding", [])
        self.sample_count: int = data.get("sample_count", 0)
        self.total_detections: int = data.get("total_detections", 0)
        self.samples: List[Dict[str, Any]] = data.get("samples", [])
        self.created_at: datetime = data.get("created_at", datetime.utcnow())
        self.updated_at: datetime = data.get("updated_at", datetime.utcnow())
        self.last_seen: Optional[datetime] = data.get("last_seen")
        self.first_seen_camera: Optional[str] = data.get("first_seen_camera")
        self.status: str = data.get("status", "unlabeled")

    @property
    def id(self) -> Optional[str]:
        return str(self._id) if self._id else None

    @property
    def primary_image_path(self) -> Optional[str]:
        if not self.samples:
            return None
        return self.samples[-1].get("image_path") or self.samples[0].get("image_path")

    def to_dict(self) -> Dict[str, Any]:
        def _serialize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
            serialized = dict(sample)
            ts = serialized.get("timestamp")
            if isinstance(ts, datetime):
                serialized["timestamp"] = ts.isoformat()
            return serialized

        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "sample_count": self.sample_count,
            "total_detections": self.total_detections,
            "last_seen": self.last_seen.isoformat() if isinstance(self.last_seen, datetime) else self.last_seen,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "primary_image_path": self.primary_image_path,
            "samples": [_serialize_sample(sample) for sample in self.samples[-6:]],
        }

    # --- CRUD helpers ---
    @classmethod
    def get_by_id(cls, profile_id: str) -> Optional["FaceProfile"]:
        try:
            data = db.face_profiles.find_one({"_id": ObjectId(profile_id)})
            return FaceProfile(data) if data else None
        except Exception:
            return None

    @classmethod
    def get_all(cls) -> List["FaceProfile"]:
        profiles = db.face_profiles.find().sort("last_seen", -1)
        return [FaceProfile(data) for data in profiles]

    @classmethod
    def cluster_profiles(
        cls,
        threshold: float = 0.8,
        statuses: Optional[Iterable[str]] = None,
    ) -> List[List["FaceProfile"]]:
        """Group profiles whose embeddings exceed the similarity threshold."""

        query: Dict[str, Any] = {}
        if statuses:
            query["status"] = {"$in": list(statuses)}

        docs = list(db.face_profiles.find(query))
        if not docs:
            return []

        profiles = [FaceProfile(doc) for doc in docs]
        ids = [profile.id for profile in profiles]
        parents = {pid: pid for pid in ids if pid}

        def _find(pid: str) -> str:
            while parents[pid] != pid:
                parents[pid] = parents[parents[pid]]
                pid = parents[pid]
            return pid

        def _union(a: str, b: str) -> None:
            root_a = _find(a)
            root_b = _find(b)
            if root_a != root_b:
                parents[root_b] = root_a

        embeddings = {
            profile.id: cls._normalize_embedding(np.array(profile.embedding, dtype=np.float32))
            for profile in profiles
            if profile.id
        }

        profile_ids = [pid for pid in ids if pid and embeddings.get(pid) is not None]

        for idx, pid_a in enumerate(profile_ids):
            emb_a = embeddings.get(pid_a)
            if emb_a is None or emb_a.size == 0:
                continue
            for pid_b in profile_ids[idx + 1 :]:
                emb_b = embeddings.get(pid_b)
                if emb_b is None or emb_b.size == 0:
                    continue
                score = float(np.dot(emb_a, emb_b))
                if score >= threshold:
                    _union(pid_a, pid_b)

        grouped: Dict[str, List[FaceProfile]] = defaultdict(list)
        for profile in profiles:
            if not profile.id:
                continue
            root = _find(profile.id)
            grouped[root].append(profile)

        return list(grouped.values())

    @classmethod
    def auto_assimilate_unlabeled(cls, threshold: float = 0.9) -> List[Dict[str, Any]]:
        """Merge unlabeled profiles into the closest known identity when above threshold."""

        known_docs = list(db.face_profiles.find({"status": "known", "name": {"$ne": None}}))
        if not known_docs:
            logger.info("auto_assimilate_unlabeled skipped: no known profiles")
            return []

        known_profiles: Dict[str, FaceProfile] = {}
        for doc in known_docs:
            profile = FaceProfile(doc)
            if profile.id:
                known_profiles[profile.id] = profile

        if not known_profiles:
            logger.info("auto_assimilate_unlabeled skipped: known profile map empty")
            return []

        unlabeled_docs = list(db.face_profiles.find({"status": "unlabeled"}))
        results: List[Dict[str, Any]] = []

        for doc in unlabeled_docs:
            unlabeled = FaceProfile(doc)
            if not unlabeled.id or not unlabeled.embedding:
                continue

            candidate_emb = cls._normalize_embedding(np.array(unlabeled.embedding, dtype=np.float32))
            if candidate_emb.size == 0:
                continue

            best_score = -1.0
            best_known_id: Optional[str] = None

            for known_id, known_profile in known_profiles.items():
                known_emb = cls._normalize_embedding(np.array(known_profile.embedding, dtype=np.float32))
                if known_emb.size == 0:
                    continue
                score = float(np.dot(known_emb, candidate_emb))
                if score > best_score:
                    best_score = score
                    best_known_id = known_id

            if best_known_id and best_score >= threshold:
                target = known_profiles[best_known_id]
                if target.merge_from(unlabeled):
                    refreshed = FaceProfile.get_by_id(best_known_id)
                    if refreshed:
                        known_profiles[best_known_id] = refreshed
                    record = {
                        "target_face_id": best_known_id,
                        "merged_face_id": unlabeled.id,
                        "score": best_score,
                    }
                    results.append(record)
                    logger.info(
                        "Assimilated unlabeled face %s into %s (score %.3f)",
                        unlabeled.id,
                        best_known_id,
                        best_score,
                    )
                else:
                    logger.warning(
                        "Failed to merge unlabeled face %s into %s despite score %.3f",
                        unlabeled.id,
                        best_known_id,
                        best_score,
                    )

        return results

    @classmethod
    def create(
        cls,
        embedding: np.ndarray,
        image_path: Optional[str],
        camera_id: Optional[str],
        detection_time: datetime,
        detection_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> "FaceProfile":
        now = datetime.utcnow()
        normalized = cls._normalize_embedding(embedding)
        status = "known" if name else "unlabeled"
        sample_entry = cls._sample_entry(image_path, detection_time, camera_id, detection_id)
        doc = {
            "name": name,
            "status": status,
            "embedding": normalized.tolist(),
            "sample_count": 1,
            "total_detections": 1,
            "samples": [sample_entry] if sample_entry else [],
            "created_at": now,
            "updated_at": now,
            "last_seen": detection_time,
            "first_seen_camera": camera_id,
        }
        result = db.face_profiles.insert_one(doc)
        doc["_id"] = result.inserted_id
        return FaceProfile(doc)

    @classmethod
    def delete(cls, profile_id: str) -> bool:
        try:
            result = db.face_profiles.delete_one({"_id": ObjectId(profile_id)})
            return bool(result.deleted_count)
        except Exception:
            return False

    # --- Recognition helpers ---
    @classmethod
    def find_best_match(cls, embedding: np.ndarray, threshold: float) -> Tuple[Optional["FaceProfile"], float]:
        normalized = cls._normalize_embedding(embedding)
        best_score = -1.0
        best_profile_id: Optional[ObjectId] = None
        cursor = db.face_profiles.find({}, {"embedding": 1})
        for data in cursor:
            existing = np.array(data.get("embedding", []), dtype=np.float32)
            if existing.size == 0:
                continue
            score = float(np.dot(existing, normalized))
            if score > best_score:
                best_score = score
                best_profile_id = data.get("_id")
        if best_score >= threshold and best_profile_id:
            full_doc = db.face_profiles.find_one({"_id": best_profile_id})
            if full_doc:
                return FaceProfile(full_doc), best_score
        return None, best_score

    def register_match(
        self,
        embedding: np.ndarray,
        image_path: Optional[str],
        detection_time: datetime,
        camera_id: Optional[str],
        detection_id: Optional[str],
    ) -> None:
        normalized = self._blend_embedding(embedding)
        sample_entry = self._sample_entry(image_path, detection_time, camera_id, detection_id)
        update: Dict[str, Any] = {
            "embedding": normalized.tolist(),
            "sample_count": self.sample_count + 1,
            "total_detections": self.total_detections + 1,
            "updated_at": datetime.utcnow(),
            "last_seen": detection_time,
        }
        if sample_entry:
            db.face_profiles.update_one(
                {"_id": self._id},
                {
                    "$set": update,
                    "$push": {
                        "samples": {
                            "$each": [sample_entry],
                            "$slice": -10,
                        }
                    },
                },
            )
            self.samples.append(sample_entry)
            if len(self.samples) > 10:
                self.samples = self.samples[-10:]
        else:
            db.face_profiles.update_one({"_id": self._id}, {"$set": update})
        self.embedding = normalized.tolist()
        self.sample_count += 1
        self.total_detections += 1
        self.last_seen = detection_time
        self.updated_at = update["updated_at"]

    def update_name(self, name: Optional[str]) -> None:
        self.name = name or None
        self.status = "known" if self.name else "unlabeled"
        self.updated_at = datetime.utcnow()
        db.face_profiles.update_one(
            {"_id": self._id},
            {
                "$set": {
                    "name": self.name,
                    "status": self.status,
                    "updated_at": self.updated_at,
                }
            },
        )

    def prune_missing_samples(self, base_path: Optional[str] = None) -> None:
        """Drop sample records whose backing images are gone."""
        if not self.samples:
            return

        if base_path is None:
            base_path = os.getcwd()

        valid_samples: List[Dict[str, Any]] = []
        for sample in self.samples:
            path = sample.get("image_path")
            if not path:
                continue
            resolved = path if os.path.isabs(path) else os.path.normpath(os.path.join(base_path, path))
            if os.path.exists(resolved):
                valid_samples.append(sample)

        if len(valid_samples) == len(self.samples):
            return

        db.face_profiles.update_one(
            {"_id": self._id},
            {
                "$set": {
                    "samples": valid_samples,
                    "sample_count": len(valid_samples),
                }
            },
        )
        self.samples = valid_samples
        self.sample_count = len(valid_samples)

    def merge_from(self, other: "FaceProfile") -> bool:
        """Merge another profile into this one, updating embeddings and samples."""
        if not other or not other.id or not self.id or other.id == self.id:
            return False

        try:
            current_emb = np.array(self.embedding, dtype=np.float32)
            other_emb = np.array(other.embedding, dtype=np.float32)

            weight_self = max(1, int(self.sample_count) if self.sample_count is not None else 0)
            weight_other = max(1, int(other.sample_count) if other.sample_count is not None else 0)

            self.status = "known" if self.name else "unlabeled"

            if current_emb.size == 0 and other_emb.size == 0:
                fused_embedding = current_emb
            elif current_emb.size == 0:
                fused_embedding = other_emb
            elif other_emb.size == 0:
                fused_embedding = current_emb
            else:
                fused_embedding = (current_emb * weight_self + other_emb * weight_other) / (weight_self + weight_other)
            fused_embedding = self._normalize_embedding(fused_embedding)

            combined_samples = list(self.samples or []) + list(other.samples or [])

            def _timestamp_key(sample: Dict[str, Any]) -> datetime:
                ts = sample.get("timestamp")
                if isinstance(ts, datetime):
                    return ts
                if isinstance(ts, str):
                    try:
                        return datetime.fromisoformat(ts)
                    except ValueError:
                        try:
                            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except ValueError:
                            pass
                return datetime.min

            combined_samples.sort(key=_timestamp_key)
            trimmed_samples = combined_samples[-10:]

            total_samples = (self.sample_count or 0) + (other.sample_count or 0)
            total_detections = (self.total_detections or 0) + (other.total_detections or 0)
            last_seen_candidates = [value for value in [self.last_seen, other.last_seen] if value]
            last_seen = max(last_seen_candidates) if last_seen_candidates else None

            update_fields: Dict[str, Any] = {
                "embedding": fused_embedding.tolist(),
                "samples": trimmed_samples,
                "sample_count": max(total_samples, len(trimmed_samples)),
                "total_detections": total_detections,
                "updated_at": datetime.utcnow(),
                "status": self.status,
            }
            if last_seen:
                update_fields["last_seen"] = last_seen

            db.face_profiles.update_one({"_id": self._id}, {"$set": update_fields})

            effective_status = "known" if self.name else self.status or "unlabeled"

            detection_update = {
                "face_profile_id": self.id,
                "face_status": effective_status,
            }
            if self.name:
                detection_update["face_recognition_name"] = self.name
            detection_update["attributes.face_profile_id"] = self.id

            db.detections.update_many(
                {"face_profile_id": other.id},
                {"$set": detection_update}
            )

            db.face_profiles.delete_one({"_id": other._id})

            self.embedding = fused_embedding.tolist()
            self.samples = trimmed_samples
            self.sample_count = update_fields["sample_count"]
            self.total_detections = total_detections
            self.status = effective_status
            if last_seen:
                self.last_seen = last_seen
            self.updated_at = update_fields["updated_at"]

            return True
        except Exception:
            return False

    # --- Internal helpers ---
    def _blend_embedding(self, new_embedding: np.ndarray) -> np.ndarray:
        current = np.array(self.embedding, dtype=np.float32)
        if current.size == 0 or self.sample_count <= 0:
            return self._normalize_embedding(new_embedding)
        blended = (current * self.sample_count + self._normalize_embedding(new_embedding)) / (self.sample_count + 1)
        return self._normalize_embedding(blended)

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    @staticmethod
    def _sample_entry(
        image_path: Optional[str],
        timestamp: datetime,
        camera_id: Optional[str],
        detection_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not image_path:
            return None
        entry = {
            "image_path": image_path,
            "timestamp": timestamp,
        }
        if camera_id:
            entry["camera_id"] = str(camera_id)
        if detection_id:
            entry["detection_id"] = str(detection_id)
        return entry