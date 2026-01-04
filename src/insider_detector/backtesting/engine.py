"""
Backtesting engine for validating detection algorithms.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from ..analysis.profiler import AccountProfiler
from ..analysis.patterns import PatternAnalyzer
from ..config import DetectionConfig, get_config
from ..models import Platform

from .metrics import (
    BacktestMetrics,
    DetectionResult,
    SignalResult,
    calculate_metrics,
    print_metrics_report,
)
from .scenarios import Scenario, get_all_scenarios, KNOWN_CASES

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Detection thresholds to test
    min_suspicion_score: float = 0.5

    # What counts as a "detection"
    require_account_match: bool = False  # If True, must detect specific accounts
    min_accounts_to_detect: int = 1  # Minimum accounts to flag

    # Signal generation
    generate_signals: bool = True
    min_signal_confidence: float = 0.5

    # Simulation settings
    simulated_bet_size: Decimal = Decimal("100")  # USD per signal


class BacktestEngine:
    """
    Engine for backtesting insider trading detection.

    Usage:
        engine = BacktestEngine()
        results = engine.run_all_scenarios()
        print(results.summary())
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        detection_config: Optional[DetectionConfig] = None,
    ):
        self.config = config or BacktestConfig()
        self.detection_config = detection_config or get_config().detection

    def run_scenario(self, scenario: Scenario) -> tuple[DetectionResult, list[SignalResult]]:
        """Run detection on a single scenario."""
        logger.info(f"Running scenario: {scenario.name}")

        # Initialize analyzers
        profiler = AccountProfiler()
        pattern_analyzer = PatternAnalyzer()

        # Add market
        pattern_analyzer.add_market(scenario.market)
        profiler.add_resolved_market(scenario.market)

        # Process all trades
        profiler.add_trades(scenario.trades)
        pattern_analyzer.add_trades(scenario.trades)

        # Get unique accounts
        accounts = set(t.account_id for t in scenario.trades)

        # Analyze each account
        detected_accounts = []
        max_score = 0.0

        for account_id in accounts:
            account_trades = [t for t in scenario.trades if t.account_id == account_id]
            profile = profiler.analyze_account(
                scenario.market.platform,
                account_id,
                account_trades,
            )

            if profile.suspicion_score > max_score:
                max_score = profile.suspicion_score

            if profile.suspicion_score >= self.config.min_suspicion_score:
                detected_accounts.append(account_id)

        # Check for pattern-based detection
        anomalies = pattern_analyzer.analyze_market(scenario.market.id)
        for anomaly in anomalies:
            for account in anomaly.accounts:
                if account not in detected_accounts:
                    detected_accounts.append(account)

        # Determine if we "detected" the insider trading
        actually_detected = len(detected_accounts) >= self.config.min_accounts_to_detect

        if self.config.require_account_match and scenario.expected_suspicious_accounts:
            # Must match specific accounts
            matched = set(detected_accounts) & set(scenario.expected_suspicious_accounts)
            actually_detected = len(matched) > 0

        # Calculate TP/FP/FN for accounts
        expected_set = set(scenario.expected_suspicious_accounts)
        detected_set = set(detected_accounts)

        true_positives = len(expected_set & detected_set)
        false_positives = len(detected_set - expected_set)
        false_negatives = len(expected_set - detected_set)

        detection_result = DetectionResult(
            scenario_id=scenario.id,
            expected_to_detect=scenario.should_detect_insider,
            actually_detected=actually_detected,
            detected_accounts=detected_accounts,
            expected_accounts=scenario.expected_suspicious_accounts,
            max_suspicion_score=max_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

        # Generate trading signals
        signal_results = []

        if self.config.generate_signals and detected_accounts:
            # What position would we take based on detected accounts?
            insider_trades = [
                t for t in scenario.trades
                if t.account_id in detected_accounts
            ]

            if insider_trades:
                # Follow the insiders - what side are they on?
                yes_volume = sum(
                    float(t.price * t.size)
                    for t in insider_trades if t.is_yes
                )
                no_volume = sum(
                    float(t.price * t.size)
                    for t in insider_trades if not t.is_yes
                )

                recommended = "YES" if yes_volume > no_volume else "NO"
                avg_price = Decimal(str(
                    sum(float(t.price) for t in insider_trades) / len(insider_trades)
                ))

                # Calculate confidence based on volume imbalance
                total_vol = yes_volume + no_volume
                confidence = abs(yes_volume - no_volume) / total_vol if total_vol > 0 else 0.5
                confidence = min(0.9, confidence + 0.3)  # Boost confidence

                if confidence >= self.config.min_signal_confidence:
                    # Calculate profit if we followed this signal
                    bet_size = self.config.simulated_bet_size
                    entry_price = avg_price

                    # If YES wins and we bet YES (or NO wins and we bet NO)
                    was_correct = (
                        (recommended == "YES" and scenario.market.outcome) or
                        (recommended == "NO" and not scenario.market.outcome)
                    )

                    if was_correct:
                        # Win: get $1 per share, minus entry price
                        shares = bet_size / entry_price
                        profit = shares * (Decimal("1") - entry_price)
                        roi = float((Decimal("1") - entry_price) / entry_price) * 100
                    else:
                        # Lose: lose entire bet
                        profit = -bet_size
                        roi = -100.0

                    signal_results.append(SignalResult(
                        scenario_id=scenario.id,
                        market_id=scenario.market.id,
                        signal_type="insider_following",
                        recommended_position=recommended,
                        confidence=confidence,
                        entry_price=entry_price,
                        actual_outcome=scenario.market.outcome,
                        profit_if_followed=profit,
                        roi_if_followed=roi,
                        was_correct=was_correct,
                    ))

        return detection_result, signal_results

    def run_all_scenarios(self) -> BacktestMetrics:
        """Run all built-in scenarios and return aggregate metrics."""
        scenarios = get_all_scenarios()
        return self.run_scenarios(scenarios)

    def run_scenarios(self, scenarios: list[Scenario]) -> BacktestMetrics:
        """Run a list of scenarios and return aggregate metrics."""
        all_detection_results = []
        all_signal_results = []

        for scenario in scenarios:
            try:
                detection, signals = self.run_scenario(scenario)
                all_detection_results.append(detection)
                all_signal_results.extend(signals)

                # Log result
                status = "✓ DETECTED" if detection.actually_detected else "✗ MISSED"
                expected = "insider" if scenario.should_detect_insider else "normal"
                logger.info(f"  {status} (expected: {expected}, score: {detection.max_suspicion_score:.2f})")

            except Exception as e:
                logger.error(f"Error running scenario {scenario.id}: {e}")

        return calculate_metrics(all_detection_results, all_signal_results)

    def optimize_thresholds(
        self,
        scenarios: list[Scenario] = None,
        suspicion_thresholds: list[float] = None,
    ) -> dict:
        """
        Run backtests with different thresholds to find optimal settings.

        Returns a dict mapping threshold to metrics.
        """
        if scenarios is None:
            scenarios = get_all_scenarios()

        if suspicion_thresholds is None:
            suspicion_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        results = {}

        for threshold in suspicion_thresholds:
            logger.info(f"Testing threshold: {threshold}")
            self.config.min_suspicion_score = threshold

            metrics = self.run_scenarios(scenarios)
            results[threshold] = {
                "threshold": threshold,
                "detection_rate": metrics.detection_rate,
                "precision": metrics.precision,
                "f1_score": metrics.f1_score,
                "signal_accuracy": metrics.signal_accuracy,
                "roi": metrics.overall_roi,
            }

        return results

    def print_report(self, metrics: BacktestMetrics) -> str:
        """Generate a formatted report."""
        return print_metrics_report(metrics)


def run_quick_backtest() -> BacktestMetrics:
    """Run a quick backtest with default settings."""
    engine = BacktestEngine()
    return engine.run_all_scenarios()


def run_and_print_backtest():
    """Run backtest and print results."""
    engine = BacktestEngine()
    metrics = engine.run_all_scenarios()
    print(engine.print_report(metrics))
    return metrics
