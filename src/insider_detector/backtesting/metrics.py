"""
Metrics for evaluating backtesting results.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from .scenarios import Scenario


@dataclass
class SignalResult:
    """Result of a single signal/trade."""
    scenario_id: str
    market_id: str
    signal_type: str
    recommended_position: str  # "YES" or "NO"
    confidence: float
    entry_price: Decimal
    actual_outcome: bool  # True = YES won
    profit_if_followed: Decimal
    roi_if_followed: float
    was_correct: bool


@dataclass
class DetectionResult:
    """Result of detection on a scenario."""
    scenario_id: str
    expected_to_detect: bool
    actually_detected: bool
    detected_accounts: list[str]
    expected_accounts: list[str]
    max_suspicion_score: float
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class BacktestMetrics:
    """Aggregate metrics from backtesting."""

    # Detection accuracy
    total_scenarios: int = 0
    scenarios_with_insiders: int = 0
    correctly_detected: int = 0
    missed_detections: int = 0
    false_alarms: int = 0

    detection_rate: float = 0.0  # TP / (TP + FN)
    precision: float = 0.0  # TP / (TP + FP)
    f1_score: float = 0.0

    # Account-level metrics
    total_insider_accounts: int = 0
    accounts_detected: int = 0
    account_detection_rate: float = 0.0

    # Trading signal performance
    total_signals: int = 0
    correct_signals: int = 0
    signal_accuracy: float = 0.0

    # Profitability (if following signals)
    total_profit: Decimal = Decimal("0")
    total_invested: Decimal = Decimal("0")
    overall_roi: float = 0.0
    win_rate: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: float = 0.0

    # Per-scenario results
    detection_results: list[DetectionResult] = field(default_factory=list)
    signal_results: list[SignalResult] = field(default_factory=list)


def calculate_metrics(
    detection_results: list[DetectionResult],
    signal_results: list[SignalResult],
) -> BacktestMetrics:
    """Calculate aggregate metrics from backtesting results."""

    metrics = BacktestMetrics()

    # Detection metrics
    metrics.total_scenarios = len(detection_results)
    metrics.scenarios_with_insiders = sum(1 for r in detection_results if r.expected_to_detect)

    tp = sum(1 for r in detection_results if r.expected_to_detect and r.actually_detected)
    fp = sum(1 for r in detection_results if not r.expected_to_detect and r.actually_detected)
    fn = sum(1 for r in detection_results if r.expected_to_detect and not r.actually_detected)

    metrics.correctly_detected = tp
    metrics.missed_detections = fn
    metrics.false_alarms = fp

    if tp + fn > 0:
        metrics.detection_rate = tp / (tp + fn)
    if tp + fp > 0:
        metrics.precision = tp / (tp + fp)
    if metrics.detection_rate + metrics.precision > 0:
        metrics.f1_score = (
            2 * metrics.detection_rate * metrics.precision /
            (metrics.detection_rate + metrics.precision)
        )

    # Account-level metrics
    all_expected = set()
    all_detected = set()
    for r in detection_results:
        all_expected.update(r.expected_accounts)
        all_detected.update(r.detected_accounts)

    metrics.total_insider_accounts = len(all_expected)
    metrics.accounts_detected = len(all_expected & all_detected)
    if metrics.total_insider_accounts > 0:
        metrics.account_detection_rate = metrics.accounts_detected / metrics.total_insider_accounts

    # Signal performance
    metrics.total_signals = len(signal_results)
    metrics.correct_signals = sum(1 for s in signal_results if s.was_correct)
    if metrics.total_signals > 0:
        metrics.signal_accuracy = metrics.correct_signals / metrics.total_signals

    # Profitability
    wins = [s for s in signal_results if s.profit_if_followed > 0]
    losses = [s for s in signal_results if s.profit_if_followed <= 0]

    metrics.total_profit = sum(s.profit_if_followed for s in signal_results)
    metrics.total_invested = sum(s.entry_price for s in signal_results)

    if metrics.total_invested > 0:
        metrics.overall_roi = float(metrics.total_profit / metrics.total_invested) * 100

    if signal_results:
        metrics.win_rate = len(wins) / len(signal_results)

    if wins:
        metrics.avg_win = sum(s.profit_if_followed for s in wins) / len(wins)
    if losses:
        metrics.avg_loss = abs(sum(s.profit_if_followed for s in losses) / len(losses))

    gross_profit = sum(s.profit_if_followed for s in wins)
    gross_loss = abs(sum(s.profit_if_followed for s in losses))
    if gross_loss > 0:
        metrics.profit_factor = float(gross_profit / gross_loss)

    metrics.detection_results = detection_results
    metrics.signal_results = signal_results

    return metrics


def print_metrics_report(metrics: BacktestMetrics) -> str:
    """Generate a formatted report of backtesting metrics."""
    lines = [
        "=" * 60,
        "BACKTESTING RESULTS",
        "=" * 60,
        "",
        "DETECTION PERFORMANCE",
        "-" * 40,
        f"Total scenarios tested: {metrics.total_scenarios}",
        f"Scenarios with insiders: {metrics.scenarios_with_insiders}",
        f"Correctly detected: {metrics.correctly_detected}",
        f"Missed detections: {metrics.missed_detections}",
        f"False alarms: {metrics.false_alarms}",
        "",
        f"Detection rate (recall): {metrics.detection_rate:.1%}",
        f"Precision: {metrics.precision:.1%}",
        f"F1 Score: {metrics.f1_score:.2f}",
        "",
        "ACCOUNT-LEVEL DETECTION",
        "-" * 40,
        f"Total insider accounts: {metrics.total_insider_accounts}",
        f"Accounts detected: {metrics.accounts_detected}",
        f"Account detection rate: {metrics.account_detection_rate:.1%}",
        "",
        "TRADING SIGNAL PERFORMANCE",
        "-" * 40,
        f"Total signals generated: {metrics.total_signals}",
        f"Correct signals: {metrics.correct_signals}",
        f"Signal accuracy: {metrics.signal_accuracy:.1%}",
        "",
        "PROFITABILITY (if following all signals)",
        "-" * 40,
        f"Total profit: ${float(metrics.total_profit):,.2f}",
        f"Total invested: ${float(metrics.total_invested):,.2f}",
        f"Overall ROI: {metrics.overall_roi:.1f}%",
        f"Win rate: {metrics.win_rate:.1%}",
        f"Average win: ${float(metrics.avg_win):,.2f}",
        f"Average loss: ${float(metrics.avg_loss):,.2f}",
        f"Profit factor: {metrics.profit_factor:.2f}",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)
