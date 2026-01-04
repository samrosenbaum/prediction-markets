"""
Insider trading detection and alerting system.
"""

from .detector import InsiderDetector
from .alerts import AlertManager

__all__ = ["InsiderDetector", "AlertManager"]
