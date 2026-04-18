"""
indicator_context.py: Helper for indicator LED lifecycle/context management.
"""
from inspection_system.app.camera_interface import IndicatorLED

def build_indicator_from_config(config: dict) -> IndicatorLED:
    led_cfg = config.get("indicator_led", {})
    return IndicatorLED(
        enabled=bool(led_cfg.get("enabled", False)),
        pass_gpio=int(led_cfg.get("pass_gpio", 23)),
        fail_gpio=int(led_cfg.get("fail_gpio", 24)),
        pulse_ms=int(led_cfg.get("pulse_ms", 750)),
    )

class IndicatorContext:
    def __init__(self, config: dict):
        self.indicator = build_indicator_from_config(config)
    def __enter__(self):
        return self.indicator
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.indicator.cleanup()
