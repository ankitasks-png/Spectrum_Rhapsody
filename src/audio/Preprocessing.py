"""Audio preprocessing: load, clean, and normalize raw recordings."""

import librosa
import numpy as np


def load_audio(file_path: str, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file and resample it.

    Args:
        file_path: Path to the audio file (.wav, .mp3, .flac, etc.)
        sample_rate: Target sample rate in Hz.

    Returns:
        Tuple of (waveform, sample_rate).
    """
    y, sr = librosa.load(file_path, sr=sample_rate)
    return y, sr


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Peak-normalize a waveform to the range [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak == 0:
        return y
    return y / peak


def trim_silence(y: np.ndarray, top_db: int = 25) -> np.ndarray:
    """Trim leading/trailing silence from a waveform."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed


def preprocess(file_path: str, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    """Full preprocessing pipeline: load, trim silence, normalize."""
    y, sr = load_audio(file_path, sample_rate)
    y = trim_silence(y)
    y = normalize_audio(y)
    return y, sr
