"""
Trade logging for tracking all trading activity.
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from ..models import Platform

logger = logging.getLogger(__name__)


@dataclass
class LoggedTrade:
    """A logged trade record."""
    id: str
    timestamp: datetime
    platform: Platform
    market_id: str
    market_title: str

    action: str  # "BUY" or "SELL"
    side: str  # "YES" or "NO"
    shares: Decimal
    price: Decimal
    total_value: Decimal

    # Optional metadata
    signal_source: str = ""
    signal_confidence: float = 0.0
    notes: str = ""

    # Fees (if applicable)
    fees: Decimal = Decimal("0")


class TradeLog:
    """
    Log all trades for record-keeping and analysis.

    Maintains a persistent log of all trades made,
    useful for tax reporting, performance analysis, and auditing.
    """

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file or Path("data/trade_log.json")
        self.trades: list[LoggedTrade] = []
        self._load()

    def _load(self) -> None:
        """Load trades from file."""
        if self.log_file.exists():
            try:
                with open(self.log_file) as f:
                    data = json.load(f)

                for trade_data in data.get("trades", []):
                    trade = LoggedTrade(
                        id=trade_data["id"],
                        timestamp=datetime.fromisoformat(trade_data["timestamp"]),
                        platform=Platform(trade_data["platform"]),
                        market_id=trade_data["market_id"],
                        market_title=trade_data["market_title"],
                        action=trade_data["action"],
                        side=trade_data["side"],
                        shares=Decimal(trade_data["shares"]),
                        price=Decimal(trade_data["price"]),
                        total_value=Decimal(trade_data["total_value"]),
                        signal_source=trade_data.get("signal_source", ""),
                        signal_confidence=trade_data.get("signal_confidence", 0.0),
                        notes=trade_data.get("notes", ""),
                        fees=Decimal(trade_data.get("fees", "0")),
                    )
                    self.trades.append(trade)

                logger.info(f"Loaded {len(self.trades)} trades from {self.log_file}")

            except Exception as e:
                logger.error(f"Failed to load trade log: {e}")

    def _save(self) -> None:
        """Save trades to file."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "trades": [
                    {
                        "id": t.id,
                        "timestamp": t.timestamp.isoformat(),
                        "platform": t.platform.value,
                        "market_id": t.market_id,
                        "market_title": t.market_title,
                        "action": t.action,
                        "side": t.side,
                        "shares": str(t.shares),
                        "price": str(t.price),
                        "total_value": str(t.total_value),
                        "signal_source": t.signal_source,
                        "signal_confidence": t.signal_confidence,
                        "notes": t.notes,
                        "fees": str(t.fees),
                    }
                    for t in self.trades
                ],
                "updated_at": datetime.now().isoformat(),
            }

            with open(self.log_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save trade log: {e}")

    def log_trade(
        self,
        platform: Platform,
        market_id: str,
        market_title: str,
        action: str,
        side: str,
        shares: Decimal,
        price: Decimal,
        signal_source: str = "",
        signal_confidence: float = 0.0,
        notes: str = "",
        fees: Decimal = Decimal("0"),
    ) -> LoggedTrade:
        """Log a new trade."""
        trade = LoggedTrade(
            id=f"{platform.value}_{market_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            platform=platform,
            market_id=market_id,
            market_title=market_title,
            action=action,
            side=side,
            shares=shares,
            price=price,
            total_value=shares * price,
            signal_source=signal_source,
            signal_confidence=signal_confidence,
            notes=notes,
            fees=fees,
        )

        self.trades.append(trade)
        self._save()

        logger.info(f"Logged trade: {action} {shares} {side} @ {price} on {market_title}")

        return trade

    def get_trades(
        self,
        platform: Optional[Platform] = None,
        market_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[LoggedTrade]:
        """Get trades with optional filters."""
        trades = self.trades

        if platform:
            trades = [t for t in trades if t.platform == platform]

        if market_id:
            trades = [t for t in trades if t.market_id == market_id]

        if since:
            trades = [t for t in trades if t.timestamp >= since]

        if until:
            trades = [t for t in trades if t.timestamp <= until]

        return sorted(trades, key=lambda t: t.timestamp, reverse=True)

    def get_summary(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> dict:
        """Get summary of trading activity."""
        trades = self.get_trades(since=since, until=until)

        if not trades:
            return {"total_trades": 0}

        buys = [t for t in trades if t.action == "BUY"]
        sells = [t for t in trades if t.action == "SELL"]

        return {
            "total_trades": len(trades),
            "buys": len(buys),
            "sells": len(sells),
            "total_volume": sum(float(t.total_value) for t in trades),
            "total_fees": sum(float(t.fees) for t in trades),
            "by_platform": {
                platform.value: len([t for t in trades if t.platform == platform])
                for platform in Platform
            },
            "by_source": {
                source: len([t for t in trades if t.signal_source == source])
                for source in set(t.signal_source for t in trades)
            },
            "first_trade": min(trades, key=lambda t: t.timestamp).timestamp.isoformat(),
            "last_trade": max(trades, key=lambda t: t.timestamp).timestamp.isoformat(),
        }

    def export_csv(self, filepath: Path) -> None:
        """Export trades to CSV for tax reporting or analysis."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "ID",
                "Timestamp",
                "Platform",
                "Market ID",
                "Market Title",
                "Action",
                "Side",
                "Shares",
                "Price",
                "Total Value",
                "Fees",
                "Signal Source",
                "Signal Confidence",
                "Notes",
            ])

            # Data
            for t in sorted(self.trades, key=lambda x: x.timestamp):
                writer.writerow([
                    t.id,
                    t.timestamp.isoformat(),
                    t.platform.value,
                    t.market_id,
                    t.market_title,
                    t.action,
                    t.side,
                    str(t.shares),
                    str(t.price),
                    str(t.total_value),
                    str(t.fees),
                    t.signal_source,
                    t.signal_confidence,
                    t.notes,
                ])

        logger.info(f"Exported {len(self.trades)} trades to {filepath}")
