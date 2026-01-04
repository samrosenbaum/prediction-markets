"""
Backtesting module for validating insider trading detection.

This module allows you to:
1. Simulate historical trading scenarios
2. Test detection algorithms against known insider cases
3. Measure signal accuracy and profitability
4. Optimize detection thresholds
"""

from .engine import BacktestEngine
from .scenarios import (
    Scenario,
    KnownInsiderCase,
    KNOWN_CASES,
    create_scenario_from_trades,
)
from .metrics import BacktestMetrics, calculate_metrics

__all__ = [
    "BacktestEngine",
    "Scenario",
    "KnownInsiderCase",
    "KNOWN_CASES",
    "create_scenario_from_trades",
    "BacktestMetrics",
    "calculate_metrics",
]
