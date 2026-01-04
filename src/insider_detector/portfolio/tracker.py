"""
Portfolio tracking for prediction market positions.
"""

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
class Position:
    """A position in a prediction market."""
    market_id: str
    market_title: str
    platform: Platform

    # Position details
    side: str  # "YES" or "NO"
    shares: Decimal
    avg_entry_price: Decimal
    total_cost: Decimal

    # Current state
    current_price: Decimal = Decimal("0.5")
    is_open: bool = True

    # Resolution
    outcome: Optional[bool] = None  # True = YES won, False = NO won
    resolved_at: Optional[datetime] = None
    realized_pnl: Decimal = Decimal("0")

    # Metadata
    opened_at: datetime = field(default_factory=datetime.now)
    notes: str = ""
    signal_source: str = ""  # e.g., "insider_detector", "manual"

    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L at current price."""
        if not self.is_open:
            return Decimal("0")

        if self.side == "YES":
            current_value = self.shares * self.current_price
        else:
            current_value = self.shares * (Decimal("1") - self.current_price)

        return current_value - self.total_cost

    @property
    def unrealized_pnl_percent(self) -> float:
        """Unrealized P&L as percentage."""
        if self.total_cost == 0:
            return 0.0
        return float(self.unrealized_pnl / self.total_cost) * 100

    @property
    def max_profit(self) -> Decimal:
        """Maximum profit if position wins."""
        return self.shares - self.total_cost

    @property
    def max_loss(self) -> Decimal:
        """Maximum loss if position loses."""
        return self.total_cost

    def close(self, outcome: bool, resolved_at: datetime = None) -> Decimal:
        """Close the position with the market outcome."""
        self.is_open = False
        self.outcome = outcome
        self.resolved_at = resolved_at or datetime.now()

        # Calculate realized P&L
        if self.side == "YES":
            if outcome:  # YES won
                self.realized_pnl = self.shares - self.total_cost
            else:  # NO won
                self.realized_pnl = -self.total_cost
        else:  # side == "NO"
            if not outcome:  # NO won
                self.realized_pnl = self.shares - self.total_cost
            else:  # YES won
                self.realized_pnl = -self.total_cost

        return self.realized_pnl


@dataclass
class PortfolioSummary:
    """Summary statistics for the portfolio."""
    total_positions: int = 0
    open_positions: int = 0
    closed_positions: int = 0

    total_invested: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")

    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0

    best_trade: Optional[Position] = None
    worst_trade: Optional[Position] = None

    # By platform
    by_platform: dict = field(default_factory=dict)

    # By signal source
    by_source: dict = field(default_factory=dict)


class PortfolioTracker:
    """
    Track your prediction market portfolio.

    Usage:
        tracker = PortfolioTracker()

        # Add a position
        tracker.add_position(
            market_id="election-2024",
            market_title="Will Biden win 2024?",
            platform=Platform.POLYMARKET,
            side="YES",
            shares=Decimal("100"),
            entry_price=Decimal("0.45"),
        )

        # Update prices
        tracker.update_price("election-2024", Decimal("0.52"))

        # Get summary
        summary = tracker.get_summary()

        # Close a position
        tracker.close_position("election-2024", outcome=True)
    """

    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = data_file or Path("data/portfolio.json")
        self.positions: dict[str, Position] = {}
        self._load()

    def _load(self) -> None:
        """Load positions from file."""
        if self.data_file.exists():
            try:
                with open(self.data_file) as f:
                    data = json.load(f)

                for pos_data in data.get("positions", []):
                    pos = Position(
                        market_id=pos_data["market_id"],
                        market_title=pos_data["market_title"],
                        platform=Platform(pos_data["platform"]),
                        side=pos_data["side"],
                        shares=Decimal(pos_data["shares"]),
                        avg_entry_price=Decimal(pos_data["avg_entry_price"]),
                        total_cost=Decimal(pos_data["total_cost"]),
                        current_price=Decimal(pos_data.get("current_price", "0.5")),
                        is_open=pos_data.get("is_open", True),
                        outcome=pos_data.get("outcome"),
                        resolved_at=datetime.fromisoformat(pos_data["resolved_at"]) if pos_data.get("resolved_at") else None,
                        realized_pnl=Decimal(pos_data.get("realized_pnl", "0")),
                        opened_at=datetime.fromisoformat(pos_data["opened_at"]) if pos_data.get("opened_at") else datetime.now(),
                        notes=pos_data.get("notes", ""),
                        signal_source=pos_data.get("signal_source", ""),
                    )
                    self.positions[pos.market_id] = pos

                logger.info(f"Loaded {len(self.positions)} positions from {self.data_file}")

            except Exception as e:
                logger.error(f"Failed to load portfolio: {e}")

    def _save(self) -> None:
        """Save positions to file."""
        try:
            self.data_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "positions": [
                    {
                        "market_id": pos.market_id,
                        "market_title": pos.market_title,
                        "platform": pos.platform.value,
                        "side": pos.side,
                        "shares": str(pos.shares),
                        "avg_entry_price": str(pos.avg_entry_price),
                        "total_cost": str(pos.total_cost),
                        "current_price": str(pos.current_price),
                        "is_open": pos.is_open,
                        "outcome": pos.outcome,
                        "resolved_at": pos.resolved_at.isoformat() if pos.resolved_at else None,
                        "realized_pnl": str(pos.realized_pnl),
                        "opened_at": pos.opened_at.isoformat(),
                        "notes": pos.notes,
                        "signal_source": pos.signal_source,
                    }
                    for pos in self.positions.values()
                ],
                "updated_at": datetime.now().isoformat(),
            }

            with open(self.data_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")

    def add_position(
        self,
        market_id: str,
        market_title: str,
        platform: Platform,
        side: str,
        shares: Decimal,
        entry_price: Decimal,
        notes: str = "",
        signal_source: str = "manual",
    ) -> Position:
        """Add a new position or update existing one."""
        total_cost = shares * entry_price

        if market_id in self.positions:
            # Update existing position
            pos = self.positions[market_id]
            if pos.side == side:
                # Same side - average in
                total_shares = pos.shares + shares
                total_investment = pos.total_cost + total_cost
                pos.avg_entry_price = total_investment / total_shares
                pos.shares = total_shares
                pos.total_cost = total_investment
            else:
                # Opposite side - reduce position
                if shares >= pos.shares:
                    # Close and flip
                    remaining = shares - pos.shares
                    pos.side = side
                    pos.shares = remaining
                    pos.avg_entry_price = entry_price
                    pos.total_cost = remaining * entry_price
                else:
                    pos.shares -= shares
                    pos.total_cost = pos.shares * pos.avg_entry_price
        else:
            # New position
            pos = Position(
                market_id=market_id,
                market_title=market_title,
                platform=platform,
                side=side,
                shares=shares,
                avg_entry_price=entry_price,
                total_cost=total_cost,
                notes=notes,
                signal_source=signal_source,
            )
            self.positions[market_id] = pos

        self._save()
        return pos

    def update_price(self, market_id: str, current_price: Decimal) -> Optional[Position]:
        """Update the current price for a position."""
        if market_id in self.positions:
            self.positions[market_id].current_price = current_price
            self._save()
            return self.positions[market_id]
        return None

    def close_position(
        self,
        market_id: str,
        outcome: bool,
        resolved_at: datetime = None,
    ) -> Optional[Decimal]:
        """Close a position with the market outcome."""
        if market_id not in self.positions:
            return None

        pos = self.positions[market_id]
        pnl = pos.close(outcome, resolved_at)
        self._save()
        return pnl

    def get_position(self, market_id: str) -> Optional[Position]:
        """Get a specific position."""
        return self.positions.get(market_id)

    def get_open_positions(self) -> list[Position]:
        """Get all open positions."""
        return [p for p in self.positions.values() if p.is_open]

    def get_closed_positions(self) -> list[Position]:
        """Get all closed positions."""
        return [p for p in self.positions.values() if not p.is_open]

    def get_summary(self) -> PortfolioSummary:
        """Get portfolio summary statistics."""
        summary = PortfolioSummary()

        all_positions = list(self.positions.values())
        open_positions = [p for p in all_positions if p.is_open]
        closed_positions = [p for p in all_positions if not p.is_open]

        summary.total_positions = len(all_positions)
        summary.open_positions = len(open_positions)
        summary.closed_positions = len(closed_positions)

        # Open positions stats
        summary.total_invested = sum(p.total_cost for p in open_positions)
        summary.current_value = sum(
            p.shares * p.current_price if p.side == "YES"
            else p.shares * (Decimal("1") - p.current_price)
            for p in open_positions
        )
        summary.unrealized_pnl = sum(p.unrealized_pnl for p in open_positions)

        # Closed positions stats
        summary.realized_pnl = sum(p.realized_pnl for p in closed_positions)
        summary.total_pnl = summary.unrealized_pnl + summary.realized_pnl

        # Win/loss
        wins = [p for p in closed_positions if p.realized_pnl > 0]
        losses = [p for p in closed_positions if p.realized_pnl <= 0]
        summary.win_count = len(wins)
        summary.loss_count = len(losses)
        if closed_positions:
            summary.win_rate = len(wins) / len(closed_positions)

        # Best/worst
        if closed_positions:
            summary.best_trade = max(closed_positions, key=lambda p: p.realized_pnl)
            summary.worst_trade = min(closed_positions, key=lambda p: p.realized_pnl)

        # By platform
        for platform in Platform:
            platform_positions = [p for p in all_positions if p.platform == platform]
            if platform_positions:
                summary.by_platform[platform.value] = {
                    "count": len(platform_positions),
                    "invested": sum(p.total_cost for p in platform_positions if p.is_open),
                    "pnl": sum(p.realized_pnl for p in platform_positions if not p.is_open),
                }

        # By signal source
        sources = set(p.signal_source for p in all_positions)
        for source in sources:
            source_positions = [p for p in all_positions if p.signal_source == source]
            closed_source = [p for p in source_positions if not p.is_open]
            if source_positions:
                summary.by_source[source or "unknown"] = {
                    "count": len(source_positions),
                    "wins": len([p for p in closed_source if p.realized_pnl > 0]),
                    "pnl": sum(p.realized_pnl for p in closed_source),
                }

        return summary

    def delete_position(self, market_id: str) -> bool:
        """Delete a position."""
        if market_id in self.positions:
            del self.positions[market_id]
            self._save()
            return True
        return False
