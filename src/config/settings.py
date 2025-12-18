import json
from pathlib import Path
from typing import Dict, Any


class Settings:

    def __init__(self, config_path: str = None):
        self.blur_threshold = 100.0
        self.resize_width = 0
        self.resize_height = 0
        self.sift_features = 2000
        self.matcher_ratio = 0.7
        self.ransac_threshold = 5.0
        self.panorama_mode = "panorama"

        self.enable_blur_filter = True
        self.enable_resize = False
        self.enable_fallback = True

        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)

    def load_from_file(self, config_path: str):
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_file(self, config_path: str):
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_config_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_')}