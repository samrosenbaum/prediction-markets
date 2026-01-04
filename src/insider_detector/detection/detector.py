"""
Main insider trading detection engine.

Combines multiple signals to identify potential insider trading:
1. Abnormal win rates on time-sensitive events
2. Large positions established shortly before major news
3. Coordinated trading across multiple accounts
4. Unusual timing patterns relative to market resolution
5. Consistent profits on low-probability outcomes
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from ..analysis.patterns import AnomalySignal, PatternAnalyzer
from ..analysis.profiler import AccountProfiler, TradingPattern
from ..clients.base import BaseClient
from ..clients.kalshi import KalshiClient
from ..clients.polymarket import PolymarketClient
from ..config import DetectionConfig, get_config
from ..models import AccountProfile, Alert, Market, Platform, SuspiciousActivity, Trade

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of running detection on a market or account."""
    suspicious_activities: list[SuspiciousActivity] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    suspicious_accounts: list[AccountProfile] = field(default_factory=list)
    anomaly_signals: list[AnomalySignal] = field(default_factory=list)
    patterns: list[TradingPattern] = field(default_factory=list)


class InsiderDetector:
    """
    Main detection engine for identifying potential insider trading.

    Monitors trades across Polymarket and Kalshi, profiles accounts,
    and generates alerts when suspicious patterns are detected.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or get_config().detection
        self.profiler = AccountProfiler()
        self.pattern_analyzer = PatternAnalyzer()

        # Caches
        self._watched_markets: dict[str, Market] = {}
        self._watched_accounts: set[str] = set()
        self._known_insiders: dict[str, AccountProfile] = {}  # Accounts flagged previously

        # Clients
        self._polymarket_client: Optional[PolymarketClient] = None
        self._kalshi_client: Optional[KalshiClient] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self) -> None:
        """Initialize API clients."""
        self._polymarket_client = PolymarketClient()
        self._kalshi_client = KalshiClient()
        await self._polymarket_client.connect()
        await self._kalshi_client.connect()

    async def close(self) -> None:
        """Close API clients."""
        if self._polymarket_client:
            await self._polymarket_client.close()
        if self._kalshi_client:
            await self._kalshi_client.close()

    def get_client(self, platform: Platform) -> BaseClient:
        """Get the appropriate client for a platform."""
        if platform == Platform.POLYMARKET:
            if not self._polymarket_client:
                raise RuntimeError("Polymarket client not connected")
            return self._polymarket_client
        else:
            if not self._kalshi_client:
                raise RuntimeError("Kalshi client not connected")
            return self._kalshi_client

    async def scan_market(self, market: Market) -> DetectionResult:
        """
        Scan a market for suspicious trading activity.

        This fetches all trades, profiles accounts, and runs
        pattern detection algorithms.
        """
        result = DetectionResult()
        logger.info(f"Scanning market: {market.title} ({market.id})")

        # Get client for this platform
        client = self.get_client(market.platform)

        # Fetch trades for this market
        trades = await client.get_trades(market_id=market.id, limit=1000)

        if not trades:
            logger.info(f"No trades found for market {market.id}")
            return result

        logger.info(f"Fetched {len(trades)} trades")

        # Add to analyzers
        self.pattern_analyzer.add_market(market)
        self.pattern_analyzer.add_trades(trades)
        self.profiler.add_trades(trades)

        if market.outcome is not None:
            self.profiler.add_resolved_market(market)

        # Analyze each unique account
        unique_accounts = set(t.account_id for t in trades)
        for account_id in unique_accounts:
            account_trades = [t for t in trades if t.account_id == account_id]
            profile = self.profiler.analyze_account(market.platform, account_id, account_trades)

            # Check if suspicious
            if profile.suspicion_score >= self.config.min_severity_for_alert:
                result.suspicious_accounts.append(profile)

                # Generate suspicious activity record
                activity = self._create_suspicious_activity(profile, market, account_trades)
                if activity:
                    result.suspicious_activities.append(activity)

            # Detect trading patterns
            patterns = self.profiler.detect_patterns(market.platform, account_id)
            result.patterns.extend(patterns)

        # Run market-level anomaly detection
        anomalies = self.pattern_analyzer.analyze_market(market.id)
        result.anomaly_signals.extend(anomalies)

        # Generate alerts from anomalies
        for anomaly in anomalies:
            alert = self._anomaly_to_alert(anomaly, market)
            if alert:
                result.alerts.append(alert)

        # Sort by severity
        result.suspicious_accounts.sort(key=lambda a: a.suspicion_score, reverse=True)
        result.alerts.sort(key=lambda a: 0 if a.priority == "critical" else 1 if a.priority == "high" else 2)

        return result

    async def scan_markets(
        self,
        platforms: list[Platform] = None,
        limit: int = 50,
        status: str = "active",
    ) -> DetectionResult:
        """Scan multiple markets across platforms."""
        platforms = platforms or [Platform.POLYMARKET, Platform.KALSHI]
        combined_result = DetectionResult()

        for platform in platforms:
            try:
                client = self.get_client(platform)
                markets = await client.get_markets(status=status, limit=limit)

                for market in markets:
                    try:
                        result = await self.scan_market(market)
                        combined_result.suspicious_activities.extend(result.suspicious_activities)
                        combined_result.alerts.extend(result.alerts)
                        combined_result.suspicious_accounts.extend(result.suspicious_accounts)
                        combined_result.anomaly_signals.extend(result.anomaly_signals)
                        combined_result.patterns.extend(result.patterns)
                    except Exception as e:
                        logger.error(f"Error scanning market {market.id}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error scanning {platform.value}: {e}")
                continue

        # Deduplicate accounts
        seen_accounts = set()
        unique_accounts = []
        for acc in combined_result.suspicious_accounts:
            key = f"{acc.platform.value}:{acc.id}"
            if key not in seen_accounts:
                seen_accounts.add(key)
                unique_accounts.append(acc)
        combined_result.suspicious_accounts = unique_accounts

        return combined_result

    async def scan_account(
        self,
        platform: Platform,
        account_id: str,
    ) -> DetectionResult:
        """Deep scan of a specific account."""
        result = DetectionResult()

        client = self.get_client(platform)

        # Fetch account's positions and trades
        positions = await client.get_account_positions(account_id)
        profile = await client.get_account_profile(account_id)

        # Get trades for each position's market
        all_trades = []
        for pos in positions:
            trades = await client.get_trades(
                market_id=pos.market_id,
                account_id=account_id,
                limit=100,
            )
            all_trades.extend(trades)

        if not all_trades:
            logger.info(f"No trades found for account {account_id}")
            return result

        # Analyze
        self.profiler.add_trades(all_trades)
        profile = self.profiler.analyze_account(platform, account_id, all_trades)

        if profile.suspicion_score >= self.config.min_severity_for_alert:
            result.suspicious_accounts.append(profile)

        # Detect patterns
        patterns = self.profiler.detect_patterns(platform, account_id)
        result.patterns.extend(patterns)

        return result

    async def find_smart_money(
        self,
        platform: Platform,
        limit: int = 100,
    ) -> list[AccountProfile]:
        """
        Find accounts that consistently outperform.

        These could be:
        - Skilled traders worth following
        - Potential insiders to monitor
        """
        client = self.get_client(platform)
        top_traders = await client.get_top_traders(limit=limit)

        smart_money = []
        for trader in top_traders:
            # Filter for high performers
            if trader.win_rate >= 0.6 and float(trader.total_pnl) > 0:
                smart_money.append(trader)

        # Sort by profitability
        smart_money.sort(key=lambda t: float(t.total_pnl), reverse=True)

        return smart_money

    async def monitor_live(
        self,
        platforms: list[Platform] = None,
        market_ids: list[str] = None,
        callback=None,
    ):
        """
        Monitor live trades and generate real-time alerts.

        Args:
            platforms: Platforms to monitor
            market_ids: Specific markets to watch (optional)
            callback: Async function to call with alerts
        """
        platforms = platforms or [Platform.POLYMARKET]

        async def process_platform(platform: Platform):
            client = self.get_client(platform)

            async for trade in client.stream_trades(market_ids):
                # Add to analyzers
                self.pattern_analyzer.add_trade(trade)
                self.profiler.add_trade(trade)

                # Check for real-time signals
                await self._check_realtime_signals(trade, callback)

        # Run monitors for all platforms concurrently
        await asyncio.gather(*[process_platform(p) for p in platforms])

    async def _check_realtime_signals(self, trade: Trade, callback=None):
        """Check a trade for immediate red flags."""
        # Large trade alert
        trade_value = float(trade.price * trade.size)

        if trade_value >= self.config.whale_position_usd:
            alert = Alert(
                id=f"whale_{trade.id}",
                timestamp=trade.timestamp,
                platform=trade.platform,
                market_id=trade.market_id,
                market_title="",  # Would need to fetch
                alert_type="whale_trade",
                priority="high",
                suspicious_accounts=[trade.account_id],
                total_suspicious_volume=trade.price * trade.size,
                current_price=trade.price,
                recommended_position="YES" if trade.is_yes else "NO",
                confidence=0.5,
                reasoning=f"Large ${trade_value:,.0f} trade detected",
            )

            if callback:
                await callback(alert)

        # Check if this account is a known insider
        account_key = f"{trade.platform.value}:{trade.account_id}"
        if account_key in self._known_insiders:
            profile = self._known_insiders[account_key]
            alert = Alert(
                id=f"insider_{trade.id}",
                timestamp=trade.timestamp,
                platform=trade.platform,
                market_id=trade.market_id,
                market_title="",
                alert_type="known_insider_trade",
                priority="critical",
                suspicious_accounts=[trade.account_id],
                total_suspicious_volume=trade.price * trade.size,
                current_price=trade.price,
                recommended_position="YES" if trade.is_yes else "NO",
                confidence=profile.suspicion_score,
                reasoning=f"Known suspicious account trading (score: {profile.suspicion_score:.2f})",
            )

            if callback:
                await callback(alert)

    def _create_suspicious_activity(
        self,
        profile: AccountProfile,
        market: Market,
        trades: list[Trade],
    ) -> Optional[SuspiciousActivity]:
        """Create a suspicious activity record from a profile."""
        if not trades:
            return None

        # Calculate total position and potential profit
        net_yes = Decimal("0")
        total_cost = Decimal("0")

        for t in trades:
            if t.is_yes:
                net_yes += t.size
            else:
                net_yes -= t.size
            total_cost += t.price * t.size

        # Determine potential profit if market resolves favorably
        position_side = "YES" if net_yes > 0 else "NO"
        position_size = abs(net_yes)

        if position_side == "YES":
            potential_profit = position_size - total_cost  # If YES wins
        else:
            potential_profit = position_size - total_cost  # If NO wins

        # Determine detection type
        detection_type = "high_win_rate"
        if profile.avg_time_to_resolution < self.config.suspicious_hours_before_resolution:
            detection_type = "pre_resolution_trading"
        elif float(profile.total_pnl) > self.config.whale_profit_usd:
            detection_type = "whale_profit"

        return SuspiciousActivity(
            id=f"sus_{profile.id}_{market.id}",
            account_id=profile.id,
            platform=market.platform,
            market_id=market.id,
            market_title=market.title,
            detection_type=detection_type,
            severity=profile.suspicion_score,
            description="; ".join(profile.suspicion_reasons),
            detected_at=datetime.now(),
            trades=trades,
            total_position_size=position_size,
            potential_profit=potential_profit,
            price_before=trades[0].price if trades else Decimal("0"),
            price_after=market.current_yes_price,
        )

    def _anomaly_to_alert(self, anomaly: AnomalySignal, market: Market) -> Optional[Alert]:
        """Convert an anomaly signal to an alert."""
        priority_map = {
            (0.0, 0.4): "low",
            (0.4, 0.6): "medium",
            (0.6, 0.8): "high",
            (0.8, 1.1): "critical",
        }

        priority = "low"
        for (low, high), p in priority_map.items():
            if low <= anomaly.severity < high:
                priority = p
                break

        # Determine recommended position based on anomaly type
        recommended = ""
        confidence = anomaly.severity * 0.6  # Scale down confidence

        if anomaly.signal_type == "coordinated_trading":
            recommended = anomaly.details.get("direction", "")
            confidence = min(confidence + 0.1, 1.0)
        elif anomaly.signal_type == "whale_accumulation":
            # Follow the whales
            whale_positions = anomaly.details.get("whale_positions", [])
            if whale_positions:
                recommended = whale_positions[0].get("position", "")

        return Alert(
            id=f"alert_{anomaly.signal_type}_{anomaly.market_id}_{int(datetime.now().timestamp())}",
            timestamp=anomaly.timestamp,
            platform=market.platform,
            market_id=anomaly.market_id,
            market_title=anomaly.market_title,
            alert_type=anomaly.signal_type,
            priority=priority,
            suspicious_accounts=anomaly.accounts,
            total_suspicious_volume=Decimal(str(anomaly.details.get("total_volume", 0))),
            current_price=market.current_yes_price,
            recommended_position=recommended,
            confidence=confidence,
            reasoning=anomaly.description,
        )

    def add_known_insider(self, platform: Platform, account_id: str, profile: AccountProfile):
        """Mark an account as a known/suspected insider for monitoring."""
        key = f"{platform.value}:{account_id}"
        self._known_insiders[key] = profile

    def get_suspicious_accounts(self) -> list[AccountProfile]:
        """Get all accounts flagged as suspicious."""
        ranked = self.profiler.rank_accounts_by_suspicion()
        return [profile for profile, score in ranked if score >= self.config.min_severity_for_alert]

    def get_account_clusters(self):
        """Get clusters of potentially related accounts."""
        return self.pattern_analyzer.detect_account_clusters()
