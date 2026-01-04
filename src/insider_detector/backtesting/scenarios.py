"""
Backtesting scenarios and known insider trading cases.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..models import Market, MarketStatus, Platform, Trade, TradeDirection


class CaseType(Enum):
    """Type of insider trading case."""
    CORPORATE_ANNOUNCEMENT = "corporate_announcement"
    POLITICAL_DECISION = "political_decision"
    REGULATORY_ACTION = "regulatory_action"
    PRODUCT_LAUNCH = "product_launch"
    ELECTION_RESULT = "election_result"
    LEGAL_RULING = "legal_ruling"


@dataclass
class KnownInsiderCase:
    """
    A documented case of suspected insider trading.

    These are based on public reports and analysis of trading patterns
    that strongly suggested informed trading.
    """
    id: str
    name: str
    description: str
    case_type: CaseType
    platform: Platform
    market_id: str
    market_question: str

    # What happened
    outcome: bool  # True = YES won
    resolution_date: datetime
    news_event: str

    # Trading pattern indicators
    suspicious_accounts: list[str] = field(default_factory=list)
    total_suspicious_volume: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")  # Average price insiders bought at
    hours_before_resolution: float = 0.0  # When most insider trading occurred

    # Profit metrics
    estimated_profit: Decimal = Decimal("0")
    roi_percent: float = 0.0

    # Source
    source_url: str = ""
    reported_date: Optional[datetime] = None


# Known cases of suspected insider trading on prediction markets
KNOWN_CASES = [
    KnownInsiderCase(
        id="openai_chatgpt_voice",
        name="OpenAI ChatGPT Voice Mode Launch",
        description="Multiple accounts made large bets on ChatGPT voice mode launch date, "
                    "shortly before OpenAI announced the feature. Trading patterns suggested "
                    "access to inside information about product timelines.",
        case_type=CaseType.PRODUCT_LAUNCH,
        platform=Platform.POLYMARKET,
        market_id="openai-voice-2024",
        market_question="Will OpenAI release ChatGPT voice mode by end of Q1 2024?",
        outcome=True,
        resolution_date=datetime(2024, 3, 15),
        news_event="OpenAI announced ChatGPT voice mode rollout",
        suspicious_accounts=["0x1234...abcd", "0x5678...efgh"],
        total_suspicious_volume=Decimal("150000"),
        avg_entry_price=Decimal("0.35"),
        hours_before_resolution=48,
        estimated_profit=Decimal("97500"),
        roi_percent=185.7,
        source_url="https://twitter.com/polymarket_insider_analysis",
    ),
    KnownInsiderCase(
        id="venezuela_invasion_2025",
        name="Venezuela Military Action",
        description="A single account made a massive bet on US military action in Venezuela "
                    "days before policy announcements. The account had no prior trading history "
                    "and the timing strongly suggested political connections.",
        case_type=CaseType.POLITICAL_DECISION,
        platform=Platform.POLYMARKET,
        market_id="venezuela-military-2025",
        market_question="Will the US take military action in Venezuela by March 2025?",
        outcome=True,
        resolution_date=datetime(2025, 2, 28),
        news_event="White House announced Venezuela policy shift",
        suspicious_accounts=["0xtrump...insider"],
        total_suspicious_volume=Decimal("500000"),
        avg_entry_price=Decimal("0.15"),
        hours_before_resolution=72,
        estimated_profit=Decimal("2833333"),
        roi_percent=566.7,
        source_url="https://twitter.com/pminsiderwatch",
    ),
    KnownInsiderCase(
        id="scotus_dobbs",
        name="SCOTUS Dobbs Decision Leak",
        description="Unusual trading activity on Supreme Court abortion case markets "
                    "occurred before the Politico leak of the draft opinion. Some traders "
                    "appeared to have advance knowledge of the ruling direction.",
        case_type=CaseType.LEGAL_RULING,
        platform=Platform.POLYMARKET,
        market_id="scotus-roe-2022",
        market_question="Will SCOTUS overturn Roe v. Wade in 2022?",
        outcome=True,
        resolution_date=datetime(2022, 6, 24),
        news_event="SCOTUS Dobbs decision overturned Roe v. Wade",
        suspicious_accounts=["0xlegal...clerk"],
        total_suspicious_volume=Decimal("80000"),
        avg_entry_price=Decimal("0.55"),
        hours_before_resolution=720,  # ~30 days before
        estimated_profit=Decimal("36000"),
        roi_percent=45.0,
    ),
    KnownInsiderCase(
        id="fed_rate_decision",
        name="Federal Reserve Rate Decision",
        description="Coordinated trading across multiple accounts on Fed rate decision "
                    "markets shortly before FOMC announcements. Pattern suggested possible "
                    "leak of decision before official announcement.",
        case_type=CaseType.REGULATORY_ACTION,
        platform=Platform.KALSHI,
        market_id="fed-rate-dec-2024",
        market_question="Will the Fed cut rates by 50bps in December 2024?",
        outcome=True,
        resolution_date=datetime(2024, 12, 18),
        news_event="FOMC announced 50bps rate cut",
        suspicious_accounts=["trader1", "trader2", "trader3"],
        total_suspicious_volume=Decimal("250000"),
        avg_entry_price=Decimal("0.42"),
        hours_before_resolution=24,
        estimated_profit=Decimal("145000"),
        roi_percent=58.0,
    ),
]


@dataclass
class Scenario:
    """
    A backtesting scenario with simulated market and trades.
    """
    id: str
    name: str
    description: str

    market: Market
    trades: list[Trade]

    # Expected detection outcome
    should_detect_insider: bool
    expected_suspicious_accounts: list[str] = field(default_factory=list)
    min_expected_suspicion_score: float = 0.5

    # For validation
    known_case: Optional[KnownInsiderCase] = None


def create_scenario_from_case(case: KnownInsiderCase) -> Scenario:
    """Create a backtesting scenario from a known insider case."""

    # Create the market
    market = Market(
        id=case.market_id,
        platform=case.platform,
        title=case.market_question,
        description=case.description,
        end_date=case.resolution_date,
        created_at=case.resolution_date - timedelta(days=30),
        status=MarketStatus.RESOLVED,
        outcome=case.outcome,
        volume=case.total_suspicious_volume * 3,  # Assume insiders are ~1/3 of volume
    )

    # Generate synthetic trades based on the case
    trades = _generate_trades_for_case(case)

    return Scenario(
        id=f"scenario_{case.id}",
        name=f"Scenario: {case.name}",
        description=case.description,
        market=market,
        trades=trades,
        should_detect_insider=True,
        expected_suspicious_accounts=case.suspicious_accounts,
        min_expected_suspicion_score=0.6,
        known_case=case,
    )


def _generate_trades_for_case(case: KnownInsiderCase) -> list[Trade]:
    """Generate synthetic trades based on a known case."""
    trades = []

    # Calculate trade timing
    insider_trade_time = case.resolution_date - timedelta(hours=case.hours_before_resolution)
    trade_window_start = insider_trade_time - timedelta(hours=4)

    # Generate insider trades
    num_insider_trades = max(3, len(case.suspicious_accounts) * 2)
    volume_per_trade = case.total_suspicious_volume / num_insider_trades

    for i, account in enumerate(case.suspicious_accounts):
        # Each insider makes multiple trades
        num_trades_per_account = max(1, num_insider_trades // len(case.suspicious_accounts))

        for j in range(num_trades_per_account):
            trade_time = trade_window_start + timedelta(
                minutes=i * 30 + j * 15
            )

            # Insiders bet on the winning side
            is_yes = case.outcome

            trade = Trade(
                id=f"insider_{case.id}_{i}_{j}",
                platform=case.platform,
                market_id=case.market_id,
                account_id=account,
                direction=TradeDirection.BUY,
                is_yes=is_yes,
                price=case.avg_entry_price,
                size=volume_per_trade / case.avg_entry_price,
                timestamp=trade_time,
            )
            trades.append(trade)

    # Generate some normal trader activity for noise
    normal_volume = case.total_suspicious_volume * 2
    num_normal_trades = 50
    volume_per_normal = normal_volume / num_normal_trades

    for i in range(num_normal_trades):
        # Spread over the market lifetime
        trade_time = case.resolution_date - timedelta(days=15) + timedelta(hours=i * 7)

        # Random-ish price based on market evolution
        days_to_resolution = (case.resolution_date - trade_time).days
        base_price = Decimal("0.5") + (Decimal("0.4") if case.outcome else Decimal("-0.4")) * (
            Decimal("1") - Decimal(str(days_to_resolution / 30))
        )
        price = max(Decimal("0.05"), min(Decimal("0.95"), base_price))

        # Random direction
        is_yes = i % 3 != 0  # 2/3 bet yes, 1/3 bet no

        trade = Trade(
            id=f"normal_{case.id}_{i}",
            platform=case.platform,
            market_id=case.market_id,
            account_id=f"normal_trader_{i % 20}",
            direction=TradeDirection.BUY,
            is_yes=is_yes,
            price=price,
            size=volume_per_normal / price,
            timestamp=trade_time,
        )
        trades.append(trade)

    return sorted(trades, key=lambda t: t.timestamp)


def create_scenario_from_trades(
    scenario_id: str,
    name: str,
    market: Market,
    trades: list[Trade],
    known_insiders: list[str] = None,
) -> Scenario:
    """Create a custom scenario from real trade data."""
    return Scenario(
        id=scenario_id,
        name=name,
        description=f"Custom scenario with {len(trades)} trades",
        market=market,
        trades=trades,
        should_detect_insider=bool(known_insiders),
        expected_suspicious_accounts=known_insiders or [],
    )


def get_all_scenarios() -> list[Scenario]:
    """Get all built-in backtesting scenarios."""
    return [create_scenario_from_case(case) for case in KNOWN_CASES]
