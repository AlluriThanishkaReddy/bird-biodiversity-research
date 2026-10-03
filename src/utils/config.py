"""
Configuration loading and validation utility for acoustic biodiversity research.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigDict(dict):
    """Dictionary supporting dot-notation attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


def find_default_config_path() -> Path:
    """
    Locate the canonical configs/config.yaml file.
    
    Searches relative to this file's package root, current working directory,
    and standard project locations.
    
    Returns:
        Path to config.yaml
    """
    candidates = [
        # Relative to current file: src/utils/config.py -> ../../configs/config.yaml
        Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml",
        Path.cwd() / "configs" / "config.yaml",
        Path.cwd() / "bird-biodiversity-research" / "configs" / "config.yaml",
        Path("C:/projects/bird-biodiversity-research/configs/config.yaml"),
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
            
    raise FileNotFoundError(
        f"Unable to locate configs/config.yaml in standard locations: {[str(p) for p in candidates]}"
    )


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validate that configuration contains required sections and values.
    
    Args:
        cfg: Configuration dictionary.
        
    Raises:
        ValueError: If mandatory sections or initial research keys are missing.
    """
    required_sections = ["project", "reproducibility", "data", "audio"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing mandatory configuration section: '{section}'")
            
    # Validate reproducibility settings
    repro = cfg.get("reproducibility", {})
    if "random_seed" not in repro:
        raise ValueError("Missing 'random_seed' in reproducibility section")
        
    # Validate dataset root
    data = cfg.get("data", {})
    if "dataset_root" not in data:
        raise ValueError("Missing 'dataset_root' in data section")
        
    # Validate audio baseline parameters
    audio = cfg.get("audio", {})
    required_audio_keys = [
        "sample_rate",
        "clip_duration",
        "n_fft",
        "hop_length",
        "n_mels",
        "fmin",
        "fmax",
    ]
    for key in required_audio_keys:
        if key not in audio:
            raise ValueError(f"Missing required audio parameter: '{key}'")


def load_config(config_path: Optional[str | Path] = None) -> ConfigDict:
    """
    Load, parse, and validate YAML configuration.
    
    Args:
        config_path: Optional path to YAML configuration file.
                     If None, the default configs/config.yaml is used.
                     
    Returns:
        ConfigDict containing configuration data with dot-attribute access.
    """
    if config_path is None:
        target_path = find_default_config_path()
    else:
        target_path = Path(config_path).resolve()
        
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")
        
    with open(target_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
        
    if not isinstance(raw_config, dict):
        raise ValueError(f"Invalid YAML structure in {target_path}; expected dictionary root.")
        
    validate_config(raw_config)
    return ConfigDict(raw_config)
