"""Extract acoustic features: pitch, tone, pauses, and rhythm."""

import librosa
import numpy as np


def extract_pitch(y: np.ndarray, sr: int) -> dict:
    """Extract pitch (fundamental frequency) statistics using pYIN.

    Returns a dict with mean, std, min, max of the voiced pitch track.
    """
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

    if len(voiced_f0) == 0:
        return {"pitch_mean": 0.0, "pitch_std": 0.0, "pitch_min": 0.0, "pitch_max": 0.0}

    return {
        "pitch_mean": float(np.mean(voiced_f0)),
        "pitch_std": float(np.std(voiced_f0)),
        "pitch_min": float(np.min(voiced_f0)),
        "pitch_max": float(np.max(voiced_f0)),
    }


def extract_pauses(y: np.ndarray, sr: int, top_db: int = 25) -> dict:
    """Detect speech segments vs. silence and summarize pause behavior.

    Returns a dict with pause count, total pause duration, and average pause length.
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    total_duration = len(y) / sr

    if len(intervals) < 2:
        return {"pause_count": 0, "total_pause_time": 0.0, "avg_pause_time": 0.0}

    pause_durations = []
    for i in range(len(intervals) - 1):
        gap_start = intervals[i][1]
        gap_end = intervals[i + 1][0]
        pause_durations.append((gap_end - gap_start) / sr)

    return {
        "pause_count": len(pause_durations),
        "total_pause_time": float(sum(pause_durations)),
        "avg_pause_time": float(np.mean(pause_durations)) if pause_durations else 0.0,
        "speech_ratio": float(sum((e - s) for s, e in intervals) / sr / total_duration),
    }


def extract_rhythm(y: np.ndarray, sr: int) -> dict:
    """Extract tempo/rhythm-related features."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    return {
        "tempo": float(tempo),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
    }


def extract_all_features(y: np.ndarray, sr: int) -> dict:
    """Run the full feature extraction suite on a preprocessed waveform."""
    features = {}
    features.update(extract_pitch(y, sr))
    features.update(extract_pauses(y, sr))
    features.update(extract_rhythm(y, sr))
    return features
