"""
P&L (Profit and Loss) calculation for trading accounts.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

from ..models import Market, Platform, Position, Trade, TradeDirection

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of a trade or position in a resolved market."""
    market_id: str
    market_title: str
    platform: Platform
    position_side: str  # "YES" or "NO"
    avg_entry_price: Decimal
    size: Decimal
    invested: Decimal
    outcome: bool  # True = YES won
    pnl: Decimal
    return_pct: float
    entry_time: datetime
    resolution_time: Optional[datetime]
    time_to_resolution_hours: float


@dataclass
class PositionSnapshot:
    """Snapshot of a position at a point in time."""
    market_id: str
    yes_shares: Decimal = Decimal("0")
    no_shares: Decimal = Decimal("0")
    yes_cost_basis: Decimal = Decimal("0")
    no_cost_basis: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @property
    def net_invested(self) -> Decimal:
        return self.yes_cost_basis + self.no_cost_basis

    def avg_yes_price(self) -> Decimal:
        if self.yes_shares == 0:
            return Decimal("0")
        return self.yes_cost_basis / self.yes_shares

    def avg_no_price(self) -> Decimal:
        if self.no_shares == 0:
            return Decimal("0")
        return self.no_cost_basis / self.no_shares


class PnLCalculator:
    """Calculate profit and loss for trading accounts."""

    def __init__(self):
        # Track positions by account and market
        self._positions: dict[str, dict[str, PositionSnapshot]] = defaultdict(
            lambda: defaultdict(PositionSnapshot)
        )
        self._trade_results: dict[str, list[TradeResult]] = defaultdict(list)

    def process_trade(self, trade: Trade) -> None:
        """Process a trade and update position."""
        account_key = f"{trade.platform.value}:{trade.account_id}"
        pos = self._positions[account_key][trade.market_id]
        pos.market_id = trade.market_id

        trade_cost = trade.price * trade.size

        if trade.direction == TradeDirection.BUY:
            if trade.is_yes:
                pos.yes_shares += trade.size
                pos.yes_cost_basis += trade_cost
            else:
                pos.no_shares += trade.size
                pos.no_cost_basis += trade_cost
        else:  # SELL
            if trade.is_yes:
                if pos.yes_shares > 0:
                    # Calculate realized P&L for this sale
                    avg_price = pos.avg_yes_price()
                    shares_to_sell = min(trade.size, pos.yes_shares)
                    pnl = (trade.price - avg_price) * shares_to_sell
                    pos.realized_pnl += pnl

                    # Reduce position
                    cost_reduction = avg_price * shares_to_sell
                    pos.yes_shares -= shares_to_sell
                    pos.yes_cost_basis -= cost_reduction
            else:
                if pos.no_shares > 0:
                    avg_price = pos.avg_no_price()
                    shares_to_sell = min(trade.size, pos.no_shares)
                    pnl = (trade.price - avg_price) * shares_to_sell
                    pos.realized_pnl += pnl

                    cost_reduction = avg_price * shares_to_sell
                    pos.no_shares -= shares_to_sell
                    pos.no_cost_basis -= cost_reduction

    def process_trades(self, trades: list[Trade]) -> None:
        """Process multiple trades in chronological order."""
        # Sort by timestamp
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        for trade in sorted_trades:
            self.process_trade(trade)

    def resolve_market(
        self,
        market: Market,
        account_id: str,
        resolution_time: Optional[datetime] = None,
    ) -> Optional[TradeResult]:
        """Resolve a market and calculate final P&L for an account."""
        if market.outcome is None:
            return None  # Market not resolved

        account_key = f"{market.platform.value}:{account_id}"
        pos = self._positions[account_key].get(market.id)

        if not pos or (pos.yes_shares == 0 and pos.no_shares == 0):
            return None  # No position

        # Calculate P&L based on resolution
        # If YES wins: YES shares worth $1, NO shares worth $0
        # If NO wins: NO shares worth $1, YES shares worth $0

        if market.outcome:  # YES won
            final_value = pos.yes_shares  # YES shares each worth $1
            # NO shares worthless, so we lose the cost basis
            resolution_pnl = final_value - pos.yes_cost_basis - pos.no_cost_basis
            position_side = "YES" if pos.yes_shares >= pos.no_shares else "NO"
        else:  # NO won
            final_value = pos.no_shares  # NO shares each worth $1
            resolution_pnl = final_value - pos.yes_cost_basis - pos.no_cost_basis
            position_side = "NO" if pos.no_shares >= pos.yes_shares else "YES"

        total_pnl = resolution_pnl + pos.realized_pnl
        total_invested = pos.net_invested

        # Calculate return percentage
        return_pct = 0.0
        if total_invested > 0:
            return_pct = float(total_pnl / total_invested) * 100

        # Calculate time to resolution
        # This would need the entry time from trades
        time_to_resolution = 0.0

        result = TradeResult(
            market_id=market.id,
            market_title=market.title,
            platform=market.platform,
            position_side=position_side,
            avg_entry_price=pos.avg_yes_price() if position_side == "YES" else pos.avg_no_price(),
            size=pos.yes_shares if position_side == "YES" else pos.no_shares,
            invested=total_invested,
            outcome=market.outcome,
            pnl=total_pnl,
            return_pct=return_pct,
            entry_time=datetime.now(),  # Would need to track from trades
            resolution_time=resolution_time,
            time_to_resolution_hours=time_to_resolution,
        )

        # Store result
        self._trade_results[account_key].append(result)

        # Clear the position
        del self._positions[account_key][market.id]

        return result

    def get_position(self, platform: Platform, account_id: str, market_id: str) -> PositionSnapshot:
        """Get current position snapshot."""
        account_key = f"{platform.value}:{account_id}"
        return self._positions[account_key][market_id]

    def get_all_positions(self, platform: Platform, account_id: str) -> dict[str, PositionSnapshot]:
        """Get all positions for an account."""
        account_key = f"{platform.value}:{account_id}"
        return dict(self._positions[account_key])

    def get_trade_results(self, platform: Platform, account_id: str) -> list[TradeResult]:
        """Get all resolved trade results for an account."""
        account_key = f"{platform.value}:{account_id}"
        return self._trade_results[account_key]

    def get_account_stats(self, platform: Platform, account_id: str) -> dict:
        """Get aggregate statistics for an account."""
        results = self.get_trade_results(platform, account_id)
        positions = self.get_all_positions(platform, account_id)

        if not results:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": Decimal("0"),
                "avg_return_pct": 0.0,
                "best_trade": None,
                "worst_trade": None,
                "open_positions": len(positions),
            }

        wins = [r for r in results if r.pnl > 0]
        losses = [r for r in results if r.pnl <= 0]

        total_pnl = sum(r.pnl for r in results)
        avg_return = sum(r.return_pct for r in results) / len(results)

        best_trade = max(results, key=lambda r: r.pnl)
        worst_trade = min(results, key=lambda r: r.pnl)

        return {
            "total_trades": len(results),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(results) if results else 0.0,
            "total_pnl": total_pnl,
            "avg_return_pct": avg_return,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "open_positions": len(positions),
            "total_invested": sum(p.net_invested for p in positions.values()),
        }

    def calculate_unrealized_pnl(
        self,
        platform: Platform,
        account_id: str,
        market_prices: dict[str, tuple[Decimal, Decimal]],  # market_id -> (yes_price, no_price)
    ) -> Decimal:
        """Calculate unrealized P&L based on current market prices."""
        positions = self.get_all_positions(platform, account_id)
        total_unrealized = Decimal("0")

        for market_id, pos in positions.items():
            if market_id not in market_prices:
                continue

            yes_price, no_price = market_prices[market_id]

            # Value of YES position at current price
            yes_value = pos.yes_shares * yes_price
            yes_unrealized = yes_value - pos.yes_cost_basis

            # Value of NO position at current price
            no_value = pos.no_shares * no_price
            no_unrealized = no_value - pos.no_cost_basis

            total_unrealized += yes_unrealized + no_unrealized

        return total_unrealized
