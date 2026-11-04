"""Small formatting helpers shared across modules."""

from __future__ import annotations

from datetime import datetime


def now_iso() -> str:
    """Return the current local time as an ISO string with second precision."""
    return datetime.now().isoformat(timespec="seconds")


def human_bytes(num_bytes: float) -> str:
    """Format a byte count as a short human-readable string (e.g. ``1.5 MB``)."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"  # pragma: no cover - loop always returns


def pct(part: float, whole: float, digits: int = 1) -> float:
    """Return ``part / whole * 100`` rounded, or 0.0 when ``whole`` is zero."""
    if not whole:
        return 0.0
    return round(100.0 * part / whole, digits)


def safe_filename(name: str, default: str = "dataset") -> str:
    """Reduce an arbitrary dataset name to a safe file stem."""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem).strip("_")
    return cleaned or default
