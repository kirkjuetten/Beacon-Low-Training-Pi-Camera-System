#!/usr/bin/env python3
"""Shared helpers for training label normalization and record class resolution."""

from __future__ import annotations


def default_final_class(feedback: str) -> str | None:
    normalized = str(feedback).strip().lower()
    if normalized == "approve":
        return "good"
    if normalized == "reject":
        return "reject"
    return None


def resolve_learning_class(record: dict) -> str | None:
    final_class = record.get("final_class")
    if final_class in {"good", "reject"}:
        return final_class
    return default_final_class(record.get("feedback", ""))