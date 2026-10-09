import numpy as np
import soundfile as sf
import pytest

from src.audio.processor import AudioConfig, AudioProcessor


@pytest.fixture
def processor():
    return AudioProcessor(AudioConfig())


def create_audio(path, duration, sample_rate=32000, channels=1):
    samples = int(duration * sample_rate)
    rng = np.random.default_rng(42)

    if channels == 1:
        audio = rng.normal(0, 0.01, samples).astype(np.float32)
    else:
        audio = rng.normal(
            0, 0.01, (samples, channels)
        ).astype(np.float32)

    sf.write(path, audio, sample_rate)


def test_short_audio_is_padded(tmp_path, processor):
    path = tmp_path / "short.wav"
    create_audio(path, duration=2.0)

    waveform = processor.load_audio(path)

    assert waveform.shape == (
        processor.config.target_samples,
    )


def test_exact_audio_keeps_target_length(tmp_path, processor):
    path = tmp_path / "exact.wav"
    create_audio(path, duration=5.0)

    waveform = processor.load_audio(path)

    assert waveform.shape == (
        processor.config.target_samples,
    )


def test_long_audio_is_cropped(tmp_path, processor):
    path = tmp_path / "long.wav"
    create_audio(path, duration=10.0)

    waveform = processor.load_audio(path)

    assert waveform.shape == (
        processor.config.target_samples,
    )


def test_stereo_audio_is_converted_to_mono(tmp_path, processor):
    path = tmp_path / "stereo.wav"
    create_audio(
        path,
        duration=5.0,
        channels=2,
    )

    waveform = processor.load_audio(path)

    assert waveform.ndim == 1
    assert waveform.shape == (
        processor.config.target_samples,
    )


def test_different_sample_rate_is_resampled(tmp_path, processor):
    path = tmp_path / "resample.wav"
    create_audio(
        path,
        duration=5.0,
        sample_rate=16000,
    )

    waveform = processor.load_audio(path)

    assert waveform.shape == (
        processor.config.target_samples,
    )


def test_invalid_file_raises_error(processor):
    with pytest.raises(FileNotFoundError):
        processor.load_audio(
            "this_file_does_not_exist.wav"
        )


def test_log_mel_shape(tmp_path, processor):
    path = tmp_path / "mel.wav"
    create_audio(path, duration=5.0)

    waveform, log_mel = processor.process_file(path)

    assert waveform.shape == (
        processor.config.target_samples,
    )

    assert log_mel.shape == (
        processor.config.n_mels,
        processor.config.time_frames,
    )


def test_log_mel_is_finite(tmp_path, processor):
    path = tmp_path / "finite.wav"
    create_audio(path, duration=5.0)

    _, log_mel = processor.process_file(path)

    assert np.isfinite(log_mel).all()


def test_instance_standardization(tmp_path, processor):
    path = tmp_path / "standardized.wav"
    create_audio(path, duration=5.0)

    _, log_mel = processor.process_file(path)

    assert abs(float(log_mel.mean())) < 1e-5
    assert np.isclose(
        float(log_mel.std()),
        1.0,
        atol=1e-4,
    )