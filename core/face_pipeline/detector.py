"""Compatibility shims — detector lives inside FaceEmbedder."""

from core.face_pipeline.embedder import DetectedFace, FaceEmbedder, get_face_embedder

__all__ = ["DetectedFace", "FaceEmbedder", "get_face_embedder"]
