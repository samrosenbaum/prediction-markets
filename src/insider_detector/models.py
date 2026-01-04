"""
Core data models for the insider trading detector.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class Platform(Enum):
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"


class TradeDirection(Enum):
    BUY = "buy"
    SELL = "sell"


class MarketStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    RESOLVED = "resolved"


@dataclass
class Market:
    """Represents a prediction market."""
    id: str
    platform: Platform
    title: str
    description: str
    end_date: Optional[datetime]
    created_at: datetime
    status: MarketStatus
    outcome: Optional[bool] = None  # True = Yes won, False = No won, None = unresolved
    volume: Decimal = Decimal("0")
    liquidity: Decimal = Decimal("0")
    category: str = ""
    tags: list[str] = field(default_factory=list)
    url: str = ""

    # Price history
    current_yes_price: Decimal = Decimal("0.5")
    current_no_price: Decimal = Decimal("0.5")


@dataclass
class Trade:
    """Represents a single trade on a prediction market."""
    id: str
    platform: Platform
    market_id: str
    account_id: str
    direction: TradeDirection
    is_yes: bool  # True if betting YES, False if betting NO
    price: Decimal
    size: Decimal  # Amount in USD or shares
    timestamp: datetime

    # Computed fields
    implied_probability: Decimal = field(init=False)

    def __post_init__(self):
        self.implied_probability = self.price if self.is_yes else (Decimal("1") - self.price)


@dataclass
class Position:
    """Represents an account's position in a market."""
    account_id: str
    platform: Platform
    market_id: str
    yes_shares: Decimal = Decimal("0")
    no_shares: Decimal = Decimal("0")
    avg_yes_price: Decimal = Decimal("0")
    avg_no_price: Decimal = Decimal("0")
    total_invested: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")

    @property
    def net_position(self) -> Decimal:
        """Positive = net long YES, Negative = net long NO."""
        return self.yes_shares - self.no_shares


@dataclass
class AccountProfile:
    """Profile of a trading account with statistics."""
    id: str
    platform: Platform
    address: str  # Wallet address for Polymarket

    # Trading statistics
    total_trades: int = 0
    total_markets: int = 0
    total_volume: Decimal = Decimal("0")

    # Performance metrics
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    win_rate: float = 0.0
    avg_return_per_trade: float = 0.0
    sharpe_ratio: float = 0.0

    # Timing analysis
    avg_time_to_resolution: float = 0.0  # Hours before resolution when trades occur
    pct_trades_before_news: float = 0.0  # % of trades within 24h before major price moves

    # Suspicion indicators
    suspicion_score: float = 0.0
    suspicion_reasons: list[str] = field(default_factory=list)

    # Tracking
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    positions: list[Position] = field(default_factory=list)

    # Risk-adjusted metrics
    max_drawdown: float = 0.0
    profit_factor: float = 0.0  # Gross profit / Gross loss


@dataclass
class SuspiciousActivity:
    """Represents a detected suspicious trading pattern."""
    id: str
    account_id: str
    platform: Platform
    market_id: str
    market_title: str
    detection_type: str
    severity: float  # 0-1, higher = more suspicious
    description: str
    detected_at: datetime

    # Trade details
    trades: list[Trade] = field(default_factory=list)
    total_position_size: Decimal = Decimal("0")
    potential_profit: Decimal = Decimal("0")

    # Context
    price_before: Decimal = Decimal("0")
    price_after: Decimal = Decimal("0")
    time_to_resolution: Optional[float] = None  # Hours
    news_event: str = ""


@dataclass
class Alert:
    """Real-time alert for potential trading opportunity."""
    id: str
    timestamp: datetime
    platform: Platform
    market_id: str
    market_title: str
    alert_type: str
    priority: str  # "low", "medium", "high", "critical"

    # Signal details
    suspicious_accounts: list[str] = field(default_factory=list)
    total_suspicious_volume: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    recommended_position: str = ""  # "YES" or "NO"
    confidence: float = 0.0

    # Reasoning
    reasoning: str = ""
    historical_accuracy: float = 0.0  # How often similar alerts were correct
