"""
Audio signal processing and feature extraction specifications.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AudioConfig:
    """
    Controlled audio preprocessing parameters conforming to initial research values.
    
    Any deviation from these defaults must be performed via controlled experiments.
    """
    sample_rate: int = 32000
    clip_duration: float = 5.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: float = 50.0
    fmax: float = 16000.0
    top_db: float = 80.0
    power: float = 2.0
    
    @property
    def target_samples(self) -> int:
        """Total discrete audio samples per analysis clip."""
        return int(self.sample_rate * self.clip_duration)
        
    @property
    def time_frames(self) -> int:
        """Expected number of time frames in computed Mel-spectrogram."""
        # For a standard centered STFT with 160,000 samples and hop_length 512
        return int(self.target_samples // self.hop_length) + 1
