"""
Alert management and notification system.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Callable, Optional

from ..models import Alert, Platform

logger = logging.getLogger(__name__)


@dataclass
class AlertFilter:
    """Filter criteria for alerts."""
    min_priority: str = "low"  # low, medium, high, critical
    platforms: list[Platform] = field(default_factory=list)
    market_ids: list[str] = field(default_factory=list)
    alert_types: list[str] = field(default_factory=list)
    min_confidence: float = 0.0
    min_volume: float = 0.0


class AlertManager:
    """
    Manages alerts and notifications for potential trading opportunities.

    Features:
    - Alert deduplication
    - Priority filtering
    - Multiple notification channels (console, file, webhook)
    - Alert history tracking
    """

    def __init__(
        self,
        history_file: Optional[Path] = None,
        dedupe_window_hours: float = 24,
    ):
        self.history_file = history_file or Path("data/alert_history.json")
        self.dedupe_window = timedelta(hours=dedupe_window_hours)

        self._alerts: list[Alert] = []
        self._alert_history: dict[str, datetime] = {}  # Alert signature -> last seen
        self._callbacks: list[Callable] = []

        # Priority ordering
        self._priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        # Load history
        self._load_history()

    def _load_history(self):
        """Load alert history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                    for sig, ts in data.items():
                        self._alert_history[sig] = datetime.fromisoformat(ts)
            except Exception as e:
                logger.warning(f"Failed to load alert history: {e}")

    def _save_history(self):
        """Save alert history to file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            data = {sig: ts.isoformat() for sig, ts in self._alert_history.items()}
            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save alert history: {e}")

    def _get_alert_signature(self, alert: Alert) -> str:
        """Generate a signature for deduplication."""
        return f"{alert.platform.value}:{alert.market_id}:{alert.alert_type}"

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if this alert is a duplicate of a recent one."""
        sig = self._get_alert_signature(alert)
        if sig in self._alert_history:
            last_seen = self._alert_history[sig]
            if datetime.now() - last_seen < self.dedupe_window:
                return True
        return False

    def _matches_filter(self, alert: Alert, filter: AlertFilter) -> bool:
        """Check if an alert matches a filter."""
        # Priority check
        if self._priority_order.get(alert.priority, 3) > self._priority_order.get(filter.min_priority, 3):
            return False

        # Platform check
        if filter.platforms and alert.platform not in filter.platforms:
            return False

        # Market check
        if filter.market_ids and alert.market_id not in filter.market_ids:
            return False

        # Alert type check
        if filter.alert_types and alert.alert_type not in filter.alert_types:
            return False

        # Confidence check
        if alert.confidence < filter.min_confidence:
            return False

        # Volume check
        if float(alert.total_suspicious_volume) < filter.min_volume:
            return False

        return True

    def add_callback(self, callback: Callable) -> None:
        """Add a callback to be notified of new alerts."""
        self._callbacks.append(callback)

    async def process_alert(
        self,
        alert: Alert,
        filter: Optional[AlertFilter] = None,
    ) -> bool:
        """
        Process an incoming alert.

        Returns True if alert was processed, False if filtered/deduplicated.
        """
        # Check for duplicates
        if self._is_duplicate(alert):
            logger.debug(f"Skipping duplicate alert: {alert.id}")
            return False

        # Apply filter
        if filter and not self._matches_filter(alert, filter):
            logger.debug(f"Alert filtered out: {alert.id}")
            return False

        # Store alert
        self._alerts.append(alert)
        self._alert_history[self._get_alert_signature(alert)] = datetime.now()

        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        # Save history
        self._save_history()

        return True

    def get_alerts(
        self,
        filter: Optional[AlertFilter] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> list[Alert]:
        """Get alerts matching criteria."""
        alerts = self._alerts

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        if filter:
            alerts = [a for a in alerts if self._matches_filter(a, filter)]

        # Sort by priority then timestamp
        alerts.sort(key=lambda a: (self._priority_order.get(a.priority, 3), -a.timestamp.timestamp()))

        return alerts[:limit]

    def get_trading_opportunities(
        self,
        min_confidence: float = 0.5,
        platforms: list[Platform] = None,
    ) -> list[dict]:
        """
        Get current trading opportunities based on alerts.

        Returns actionable recommendations.
        """
        opportunities = []

        filter = AlertFilter(
            min_priority="medium",
            min_confidence=min_confidence,
            platforms=platforms or [],
        )

        recent_alerts = self.get_alerts(
            filter=filter,
            since=datetime.now() - timedelta(hours=24),
        )

        for alert in recent_alerts:
            if not alert.recommended_position:
                continue

            opportunities.append({
                "market_id": alert.market_id,
                "market_title": alert.market_title,
                "platform": alert.platform.value,
                "recommended_position": alert.recommended_position,
                "confidence": alert.confidence,
                "current_price": float(alert.current_price),
                "reasoning": alert.reasoning,
                "suspicious_accounts": alert.suspicious_accounts,
                "volume": float(alert.total_suspicious_volume),
                "priority": alert.priority,
                "timestamp": alert.timestamp.isoformat(),
            })

        # Sort by confidence
        opportunities.sort(key=lambda o: o["confidence"], reverse=True)

        return opportunities

    def format_alert(self, alert: Alert) -> str:
        """Format an alert for display."""
        priority_emoji = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "📢",
            "low": "📌",
        }

        emoji = priority_emoji.get(alert.priority, "📌")

        lines = [
            f"{emoji} [{alert.priority.upper()}] {alert.alert_type}",
            f"Market: {alert.market_title}",
            f"Platform: {alert.platform.value}",
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"Reasoning: {alert.reasoning}",
        ]

        if alert.recommended_position:
            lines.append(f"Recommendation: {alert.recommended_position} @ {float(alert.current_price):.2%}")
            lines.append(f"Confidence: {alert.confidence:.1%}")

        if alert.suspicious_accounts:
            lines.append(f"Suspicious accounts: {len(alert.suspicious_accounts)}")

        if alert.total_suspicious_volume > 0:
            lines.append(f"Volume: ${float(alert.total_suspicious_volume):,.2f}")

        return "\n".join(lines)

    def export_alerts(self, filepath: Path) -> None:
        """Export alerts to JSON file."""
        def serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return str(obj)
            if isinstance(obj, Platform):
                return obj.value
            return str(obj)

        alerts_data = []
        for alert in self._alerts:
            alert_dict = {
                "id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "platform": alert.platform.value,
                "market_id": alert.market_id,
                "market_title": alert.market_title,
                "alert_type": alert.alert_type,
                "priority": alert.priority,
                "suspicious_accounts": alert.suspicious_accounts,
                "total_suspicious_volume": str(alert.total_suspicious_volume),
                "current_price": str(alert.current_price),
                "recommended_position": alert.recommended_position,
                "confidence": alert.confidence,
                "reasoning": alert.reasoning,
            }
            alerts_data.append(alert_dict)

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(alerts_data, f, indent=2, default=serialize)

    def clear_old_alerts(self, max_age_hours: float = 168) -> int:
        """Clear alerts older than max_age_hours. Returns count removed."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        original_count = len(self._alerts)
        self._alerts = [a for a in self._alerts if a.timestamp >= cutoff]

        # Also clean history
        self._alert_history = {
            sig: ts for sig, ts in self._alert_history.items()
            if ts >= cutoff
        }
        self._save_history()

        return original_count - len(self._alerts)


class ConsoleNotifier:
    """Print alerts to console."""

    def __init__(self, alert_manager: AlertManager):
        self.manager = alert_manager
        alert_manager.add_callback(self._notify)

    def _notify(self, alert: Alert):
        """Print alert to console."""
        print("\n" + "=" * 60)
        print(self.manager.format_alert(alert))
        print("=" * 60 + "\n")


class WebhookNotifier:
    """Send alerts to a webhook URL."""

    def __init__(
        self,
        alert_manager: AlertManager,
        webhook_url: str,
        min_priority: str = "medium",
    ):
        self.manager = alert_manager
        self.webhook_url = webhook_url
        self.min_priority = min_priority
        self._priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        alert_manager.add_callback(self._notify)

    async def _notify(self, alert: Alert):
        """Send alert to webhook."""
        import httpx

        # Priority filter
        if self._priority_order.get(alert.priority, 3) > self._priority_order.get(self.min_priority, 3):
            return

        payload = {
            "alert_type": alert.alert_type,
            "priority": alert.priority,
            "market": alert.market_title,
            "platform": alert.platform.value,
            "reasoning": alert.reasoning,
            "recommendation": alert.recommended_position,
            "confidence": alert.confidence,
            "timestamp": alert.timestamp.isoformat(),
        }

        try:
            async with httpx.AsyncClient() as client:
                await client.post(self.webhook_url, json=payload)
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
