"""Configuration settings for the balloon panorama processor."""
import json
from pathlib import Path
from typing import Dict, Any


class Settings:
    """Configuration settings for the balloon panorama processor."""
    
    def __init__(self, config_path: str = None):
        """Initialize settings with default values or from config file."""
        # Default settings
        self.blur_threshold = 100.0
        self.resize_width = 800
        self.resize_height = 600
        self.sift_features = 2000
        self.matcher_ratio = 0.7
        self.ransac_threshold = 5.0
        self.panorama_mode = "panorama"  # or "scans"
        
        # Process settings
        self.enable_blur_filter = True
        self.enable_resize = True
        self.enable_fallback = True
        
        # Load from config file if provided
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load settings from a JSON configuration file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_to_file(self, config_path: str):
        """Save current settings to a JSON configuration file."""
        config_dict = {
            key: value for key, value in self.__dict__.items() 
            if not key.startswith('_')
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Return settings as a dictionary."""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}