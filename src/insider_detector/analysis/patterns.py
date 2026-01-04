"""
Advanced pattern analysis for detecting suspicious trading behavior.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np
from scipy import stats

from ..config import get_config
from ..models import Market, Platform, Trade

logger = logging.getLogger(__name__)


@dataclass
class AnomalySignal:
    """A detected anomaly in trading data."""
    signal_type: str
    severity: float  # 0-1
    market_id: str
    market_title: str
    accounts: list[str]
    timestamp: datetime
    description: str
    details: dict = field(default_factory=dict)


@dataclass
class AccountCluster:
    """A cluster of potentially related accounts."""
    accounts: list[str]
    correlation: float
    shared_markets: list[str]
    timing_similarity: float
    size_similarity: float
    description: str


class PatternAnalyzer:
    """Analyze trading patterns to detect insider trading signals."""

    def __init__(self):
        self.config = get_config().detection
        self._trades_by_market: dict[str, list[Trade]] = defaultdict(list)
        self._trades_by_account: dict[str, list[Trade]] = defaultdict(list)
        self._markets: dict[str, Market] = {}
        self._price_history: dict[str, list[tuple[datetime, Decimal]]] = defaultdict(list)

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the analyzer."""
        self._trades_by_market[trade.market_id].append(trade)
        account_key = f"{trade.platform.value}:{trade.account_id}"
        self._trades_by_account[account_key].append(trade)

    def add_trades(self, trades: list[Trade]) -> None:
        """Add multiple trades."""
        for trade in trades:
            self.add_trade(trade)

    def add_market(self, market: Market) -> None:
        """Add market data."""
        self._markets[market.id] = market

    def add_price_point(self, market_id: str, timestamp: datetime, price: Decimal) -> None:
        """Add a price history point."""
        self._price_history[market_id].append((timestamp, price))

    def detect_pre_announcement_surge(
        self,
        market_id: str,
        window_hours: float = 24,
    ) -> Optional[AnomalySignal]:
        """Detect unusual trading volume before major price moves."""
        trades = self._trades_by_market.get(market_id, [])
        market = self._markets.get(market_id)

        if not trades or not market:
            return None

        # Sort trades by time
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        if len(sorted_trades) < 10:
            return None

        # Calculate volume in windows
        window = timedelta(hours=window_hours)
        volume_by_window: dict[datetime, Decimal] = defaultdict(Decimal)

        for trade in sorted_trades:
            # Round to window
            window_start = trade.timestamp.replace(minute=0, second=0, microsecond=0)
            volume_by_window[window_start] += trade.price * trade.size

        if len(volume_by_window) < 5:
            return None

        # Calculate z-scores for each window
        volumes = list(volume_by_window.values())
        mean_vol = np.mean([float(v) for v in volumes])
        std_vol = np.std([float(v) for v in volumes])

        if std_vol == 0:
            return None

        # Find anomalous windows
        anomalous_windows = []
        for window_time, vol in volume_by_window.items():
            z_score = (float(vol) - mean_vol) / std_vol
            if z_score > 2.5:  # More than 2.5 standard deviations
                anomalous_windows.append((window_time, vol, z_score))

        if not anomalous_windows:
            return None

        # Get accounts active during anomalous periods
        suspicious_accounts = set()
        for window_time, _, _ in anomalous_windows:
            window_end = window_time + window
            for trade in sorted_trades:
                if window_time <= trade.timestamp < window_end:
                    suspicious_accounts.add(trade.account_id)

        # Check if this preceded a big price move
        price_history = self._price_history.get(market_id, [])
        big_move = False
        if price_history:
            sorted_prices = sorted(price_history, key=lambda x: x[0])
            for window_time, _, z_score in anomalous_windows:
                # Find prices before and after
                prices_before = [p for t, p in sorted_prices if t < window_time]
                prices_after = [p for t, p in sorted_prices if t > window_time + window]

                if prices_before and prices_after:
                    price_before = float(prices_before[-1])
                    price_after = float(prices_after[0])
                    move = abs(price_after - price_before)
                    if move > 0.2:  # 20% price move
                        big_move = True

        severity = min(max(anomalous_windows, key=lambda x: x[2])[2] / 5.0, 1.0)
        if big_move:
            severity = min(severity + 0.2, 1.0)

        return AnomalySignal(
            signal_type="pre_announcement_surge",
            severity=severity,
            market_id=market_id,
            market_title=market.title if market else market_id,
            accounts=list(suspicious_accounts),
            timestamp=anomalous_windows[0][0],
            description=f"Unusual volume spike ({len(anomalous_windows)} windows) before price movement",
            details={
                "max_z_score": max(w[2] for w in anomalous_windows),
                "anomalous_windows": len(anomalous_windows),
                "big_price_move": big_move,
            },
        )

    def detect_coordinated_trading(
        self,
        market_id: str,
        time_window_minutes: int = 30,
    ) -> Optional[AnomalySignal]:
        """Detect multiple accounts trading in coordination."""
        trades = self._trades_by_market.get(market_id, [])

        if len(trades) < 10:
            return None

        # Group trades by time window
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        window = timedelta(minutes=time_window_minutes)

        clusters = []
        current_cluster: list[Trade] = []
        cluster_start = sorted_trades[0].timestamp

        for trade in sorted_trades:
            if trade.timestamp - cluster_start <= window:
                current_cluster.append(trade)
            else:
                if len(current_cluster) >= 3:
                    clusters.append(current_cluster)
                current_cluster = [trade]
                cluster_start = trade.timestamp

        if len(current_cluster) >= 3:
            clusters.append(current_cluster)

        # Find clusters with multiple unique accounts all trading same direction
        suspicious_clusters = []
        for cluster in clusters:
            accounts = set(t.account_id for t in cluster)
            if len(accounts) >= self.config.min_correlated_accounts:
                # Check if all trading same direction
                yes_trades = sum(1 for t in cluster if t.is_yes)
                same_direction = yes_trades == len(cluster) or yes_trades == 0

                if same_direction:
                    suspicious_clusters.append({
                        "trades": cluster,
                        "accounts": list(accounts),
                        "direction": "YES" if yes_trades > 0 else "NO",
                        "volume": sum(float(t.price * t.size) for t in cluster),
                    })

        if not suspicious_clusters:
            return None

        # Return the most suspicious cluster
        most_suspicious = max(suspicious_clusters, key=lambda c: len(c["accounts"]) * c["volume"])

        market = self._markets.get(market_id)

        return AnomalySignal(
            signal_type="coordinated_trading",
            severity=min(len(most_suspicious["accounts"]) / 10, 1.0),
            market_id=market_id,
            market_title=market.title if market else market_id,
            accounts=most_suspicious["accounts"],
            timestamp=most_suspicious["trades"][0].timestamp,
            description=f"{len(most_suspicious['accounts'])} accounts trading {most_suspicious['direction']} within {time_window_minutes} minutes",
            details={
                "direction": most_suspicious["direction"],
                "total_volume": most_suspicious["volume"],
                "trade_count": len(most_suspicious["trades"]),
            },
        )

    def detect_whale_accumulation(
        self,
        market_id: str,
        threshold_usd: float = 10000,
    ) -> Optional[AnomalySignal]:
        """Detect large positions being accumulated by single accounts."""
        trades = self._trades_by_market.get(market_id, [])

        if not trades:
            return None

        # Calculate position by account
        positions: dict[str, dict] = defaultdict(lambda: {"yes": Decimal("0"), "no": Decimal("0"), "cost": Decimal("0")})

        for trade in trades:
            pos = positions[trade.account_id]
            if trade.is_yes:
                pos["yes"] += trade.size
            else:
                pos["no"] += trade.size
            pos["cost"] += trade.price * trade.size

        # Find whale positions
        whales = []
        for account_id, pos in positions.items():
            cost = float(pos["cost"])
            if cost >= threshold_usd:
                whales.append({
                    "account": account_id,
                    "position": "YES" if pos["yes"] > pos["no"] else "NO",
                    "size": float(max(pos["yes"], pos["no"])),
                    "cost": cost,
                })

        if not whales:
            return None

        # Sort by cost
        whales.sort(key=lambda w: w["cost"], reverse=True)
        market = self._markets.get(market_id)

        return AnomalySignal(
            signal_type="whale_accumulation",
            severity=min(whales[0]["cost"] / (threshold_usd * 10), 1.0),
            market_id=market_id,
            market_title=market.title if market else market_id,
            accounts=[w["account"] for w in whales],
            timestamp=datetime.now(),
            description=f"{len(whales)} whale accounts with positions > ${threshold_usd:,.0f}",
            details={
                "largest_position": whales[0]["cost"],
                "total_whale_volume": sum(w["cost"] for w in whales),
                "whale_positions": whales[:5],  # Top 5
            },
        )

    def detect_account_clusters(self) -> list[AccountCluster]:
        """Find clusters of potentially related accounts."""
        clusters = []

        # Get all accounts
        accounts = list(self._trades_by_account.keys())

        if len(accounts) < 2:
            return clusters

        # Calculate similarity between account pairs
        for i, acc1 in enumerate(accounts):
            for acc2 in accounts[i + 1:]:
                trades1 = self._trades_by_account[acc1]
                trades2 = self._trades_by_account[acc2]

                # Markets in common
                markets1 = set(t.market_id for t in trades1)
                markets2 = set(t.market_id for t in trades2)
                shared = markets1 & markets2

                if len(shared) < 2:
                    continue

                # Calculate correlation on shared markets
                correlation = self._calculate_trade_correlation(trades1, trades2, shared)

                if correlation >= self.config.correlation_threshold:
                    clusters.append(AccountCluster(
                        accounts=[acc1.split(":")[-1], acc2.split(":")[-1]],
                        correlation=correlation,
                        shared_markets=list(shared),
                        timing_similarity=self._calculate_timing_similarity(trades1, trades2, shared),
                        size_similarity=self._calculate_size_similarity(trades1, trades2, shared),
                        description=f"High correlation ({correlation:.2f}) across {len(shared)} markets",
                    ))

        # Merge overlapping clusters
        return self._merge_clusters(clusters)

    def _calculate_trade_correlation(
        self,
        trades1: list[Trade],
        trades2: list[Trade],
        shared_markets: set[str],
    ) -> float:
        """Calculate correlation between two accounts' trading behavior."""
        positions1 = {}
        positions2 = {}

        for t in trades1:
            if t.market_id in shared_markets:
                positions1[t.market_id] = 1 if t.is_yes else -1

        for t in trades2:
            if t.market_id in shared_markets:
                positions2[t.market_id] = 1 if t.is_yes else -1

        # Calculate agreement rate
        agreements = 0
        for market_id in shared_markets:
            if market_id in positions1 and market_id in positions2:
                if positions1[market_id] == positions2[market_id]:
                    agreements += 1

        return agreements / len(shared_markets) if shared_markets else 0.0

    def _calculate_timing_similarity(
        self,
        trades1: list[Trade],
        trades2: list[Trade],
        shared_markets: set[str],
    ) -> float:
        """Calculate how similar the timing is between two accounts."""
        time_diffs = []

        trades1_by_market = defaultdict(list)
        trades2_by_market = defaultdict(list)

        for t in trades1:
            if t.market_id in shared_markets:
                trades1_by_market[t.market_id].append(t.timestamp)

        for t in trades2:
            if t.market_id in shared_markets:
                trades2_by_market[t.market_id].append(t.timestamp)

        for market_id in shared_markets:
            times1 = trades1_by_market.get(market_id, [])
            times2 = trades2_by_market.get(market_id, [])

            if times1 and times2:
                # Compare first trades
                diff = abs((times1[0] - times2[0]).total_seconds() / 3600)
                time_diffs.append(diff)

        if not time_diffs:
            return 0.0

        # High similarity = small time differences
        avg_diff = np.mean(time_diffs)
        # Scale: 0h diff = 1.0 similarity, 24h diff = 0.0 similarity
        return max(0, 1 - avg_diff / 24)

    def _calculate_size_similarity(
        self,
        trades1: list[Trade],
        trades2: list[Trade],
        shared_markets: set[str],
    ) -> float:
        """Calculate how similar the position sizes are."""
        sizes1 = []
        sizes2 = []

        for t in trades1:
            if t.market_id in shared_markets:
                sizes1.append(float(t.price * t.size))

        for t in trades2:
            if t.market_id in shared_markets:
                sizes2.append(float(t.price * t.size))

        if not sizes1 or not sizes2:
            return 0.0

        # Compare average sizes
        avg1 = np.mean(sizes1)
        avg2 = np.mean(sizes2)

        if avg1 == 0 and avg2 == 0:
            return 1.0

        # Ratio-based similarity
        ratio = min(avg1, avg2) / max(avg1, avg2) if max(avg1, avg2) > 0 else 0
        return ratio

    def _merge_clusters(self, clusters: list[AccountCluster]) -> list[AccountCluster]:
        """Merge clusters with overlapping accounts."""
        if not clusters:
            return []

        # Use union-find to merge
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for cluster in clusters:
            if len(cluster.accounts) >= 2:
                for acc in cluster.accounts[1:]:
                    union(cluster.accounts[0], acc)

        # Group accounts by root
        groups: dict[str, set] = defaultdict(set)
        for cluster in clusters:
            for acc in cluster.accounts:
                root = find(acc)
                groups[root].add(acc)

        # Create merged clusters
        merged = []
        for accounts in groups.values():
            if len(accounts) >= 2:
                # Find all shared markets
                all_shared = set()
                max_correlation = 0.0
                for cluster in clusters:
                    if set(cluster.accounts) <= accounts:
                        all_shared.update(cluster.shared_markets)
                        max_correlation = max(max_correlation, cluster.correlation)

                merged.append(AccountCluster(
                    accounts=list(accounts),
                    correlation=max_correlation,
                    shared_markets=list(all_shared),
                    timing_similarity=0.0,  # Would need to recalculate
                    size_similarity=0.0,
                    description=f"Cluster of {len(accounts)} potentially related accounts",
                ))

        return merged

    def analyze_market(self, market_id: str) -> list[AnomalySignal]:
        """Run all anomaly detection on a market."""
        signals = []

        surge = self.detect_pre_announcement_surge(market_id)
        if surge and surge.severity >= self.config.min_severity_for_alert:
            signals.append(surge)

        coordinated = self.detect_coordinated_trading(market_id)
        if coordinated and coordinated.severity >= self.config.min_severity_for_alert:
            signals.append(coordinated)

        whale = self.detect_whale_accumulation(market_id)
        if whale and whale.severity >= self.config.min_severity_for_alert:
            signals.append(whale)

        return signals

    def get_all_anomalies(self) -> list[AnomalySignal]:
        """Analyze all tracked markets for anomalies."""
        all_signals = []

        for market_id in self._trades_by_market.keys():
            signals = self.analyze_market(market_id)
            all_signals.extend(signals)

        # Sort by severity
        return sorted(all_signals, key=lambda s: s.severity, reverse=True)
