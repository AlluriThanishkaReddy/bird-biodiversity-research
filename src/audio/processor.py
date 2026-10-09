"""
Audio loading and log-mel spectrogram preprocessing for BirdCLEF.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np


@dataclass(frozen=True)
class AudioConfig:
    """Controlled audio preprocessing parameters."""

    sample_rate: int = 32000
    clip_duration: float = 5.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 16000.0
    top_db: float = 80.0
    power: float = 2.0
    normalization: str = "instance_standardization"

    @property
    def target_samples(self) -> int:
        """Number of samples in one fixed-length audio clip."""
        return int(self.sample_rate * self.clip_duration)

    @property
    def time_frames(self) -> int:
        """Expected number of time frames in the centered STFT."""
        return int(self.target_samples // self.hop_length) + 1


class AudioProcessor:
    """
    Reusable BirdCLEF audio preprocessing pipeline.

    Pipeline:
        audio file
        -> mono waveform
        -> resampling if necessary
        -> fixed 5-second waveform
        -> Mel spectrogram
        -> log-Mel spectrogram
        -> optional standardization
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    def load_audio(
        self,
        path: str | Path,
        start_sec: float = 0.0,
    ) -> np.ndarray:
        """
        Load an audio file and return a fixed-length mono waveform.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        if start_sec < 0:
            raise ValueError(
                "start_sec must be greater than or equal to 0"
            )

        try:
            waveform, source_sr = librosa.load(
                path,
                sr=None,
                mono=True,
            )
        except Exception as exc:
            raise ValueError(
                f"Unable to decode audio file: {path}"
            ) from exc

        if waveform.size == 0:
            raise ValueError(
                f"Audio file contains no samples: {path}"
            )

        waveform = waveform.astype(np.float32)

        # Resample only when the source sample rate differs.
        if source_sr != self.config.sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=source_sr,
                target_sr=self.config.sample_rate,
            ).astype(np.float32)

        target_samples = self.config.target_samples

        start_sample = int(
            start_sec * self.config.sample_rate
        )

        # If the requested start is beyond the recording,
        # return a fully padded clip.
        if start_sample >= len(waveform):
            return np.zeros(
                target_samples,
                dtype=np.float32,
            )

        waveform = waveform[
            start_sample:start_sample + target_samples
        ]

        # Zero-pad recordings shorter than 5 seconds.
        if len(waveform) < target_samples:
            waveform = np.pad(
                waveform,
                (0, target_samples - len(waveform)),
                mode="constant",
            )

        return waveform.astype(np.float32)

    def waveform_to_mel(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Convert a fixed-length waveform into a log-Mel spectrogram.
        """

        if waveform.ndim != 1:
            raise ValueError(
                "Expected a mono waveform with shape "
                f"(samples,), got {waveform.shape}"
            )

        if len(waveform) != self.config.target_samples:
            raise ValueError(
                f"Expected {self.config.target_samples} samples, "
                f"got {len(waveform)}"
            )

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            power=self.config.power,
        )

        log_mel = librosa.power_to_db(
            mel,
            ref=np.max,
            top_db=self.config.top_db,
        )

        log_mel = log_mel.astype(np.float32)

        if self.config.normalization == "instance_standardization":
            mean = np.mean(log_mel)
            std = np.std(log_mel)

            log_mel = (
                (log_mel - mean)
                / (std + 1e-8)
            )

        elif self.config.normalization == "none":
            pass

        else:
            raise ValueError(
                "Unsupported normalization: "
                f"{self.config.normalization}"
            )

        return log_mel.astype(np.float32)

    def process_file(
        self,
        path: str | Path,
        start_sec: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load an audio file and return waveform and log-Mel features.
        """

        waveform = self.load_audio(
            path,
            start_sec=start_sec,
        )

        log_mel = self.waveform_to_mel(
            waveform
        )

        return waveform, log_mel

    def expected_mel_shape(self) -> Tuple[int, int]:
        """Return expected (Mel bins, time frames)."""

        return (
            self.config.n_mels,
            self.config.time_frames,
        )