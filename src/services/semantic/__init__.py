"""Semantic processing service (Dual-Model: Nomic primary + BGE fallback)."""
from services.semantic.engine import SemanticEngine, get_semantic_engine

__all__ = ["SemanticEngine", "get_semantic_engine"]

