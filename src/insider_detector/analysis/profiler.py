"""
Account profiler for analyzing trading behavior and performance.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np

from ..config import get_config
from ..models import AccountProfile, Market, Platform, Trade

logger = logging.getLogger(__name__)


@dataclass
class TradingPattern:
    """Detected trading pattern for an account."""
    pattern_type: str
    confidence: float
    description: str
    evidence: list[str]
    affected_markets: list[str]


class AccountProfiler:
    """Profile trading accounts to detect patterns and calculate metrics."""

    def __init__(self):
        self.config = get_config().detection
        self._trades_by_account: dict[str, list[Trade]] = defaultdict(list)
        self._resolved_markets: dict[str, Market] = {}
        self._profiles: dict[str, AccountProfile] = {}

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the profiler."""
        account_key = f"{trade.platform.value}:{trade.account_id}"
        self._trades_by_account[account_key].append(trade)

    def add_trades(self, trades: list[Trade]) -> None:
        """Add multiple trades."""
        for trade in trades:
            self.add_trade(trade)

    def add_resolved_market(self, market: Market) -> None:
        """Add a resolved market for P&L calculation."""
        if market.outcome is not None:
            self._resolved_markets[f"{market.platform.value}:{market.id}"] = market

    def _get_account_key(self, platform: Platform, account_id: str) -> str:
        return f"{platform.value}:{account_id}"

    def get_trades(self, platform: Platform, account_id: str) -> list[Trade]:
        """Get all trades for an account."""
        return self._trades_by_account[self._get_account_key(platform, account_id)]

    def analyze_account(
        self,
        platform: Platform,
        account_id: str,
        trades: Optional[list[Trade]] = None,
    ) -> AccountProfile:
        """Analyze an account and build a profile."""
        account_key = self._get_account_key(platform, account_id)

        if trades:
            self._trades_by_account[account_key] = trades

        account_trades = self._trades_by_account[account_key]

        if not account_trades:
            return AccountProfile(
                id=account_id,
                platform=platform,
                address=account_id,
            )

        # Sort trades by time
        sorted_trades = sorted(account_trades, key=lambda t: t.timestamp)

        # Basic stats
        total_volume = sum(t.price * t.size for t in sorted_trades)
        unique_markets = set(t.market_id for t in sorted_trades)

        # Calculate win rate from resolved markets
        wins = 0
        losses = 0
        total_pnl = Decimal("0")
        pnl_list: list[float] = []

        markets_traded = defaultdict(list)
        for trade in sorted_trades:
            markets_traded[trade.market_id].append(trade)

        for market_id, market_trades in markets_traded.items():
            market_key = f"{platform.value}:{market_id}"
            market = self._resolved_markets.get(market_key)

            if market and market.outcome is not None:
                # Calculate position from trades
                net_yes = Decimal("0")
                net_cost = Decimal("0")

                for t in market_trades:
                    if t.is_yes:
                        net_yes += t.size
                        net_cost += t.price * t.size
                    else:
                        net_yes -= t.size
                        net_cost += t.price * t.size

                # Simple P&L calc (ignoring partial exits for now)
                if net_yes > 0:  # Long YES
                    if market.outcome:  # YES won
                        pnl = net_yes - net_cost  # Each YES share worth $1
                        wins += 1
                    else:
                        pnl = -net_cost  # Lost the investment
                        losses += 1
                elif net_yes < 0:  # Long NO
                    if not market.outcome:  # NO won
                        pnl = abs(net_yes) - net_cost
                        wins += 1
                    else:
                        pnl = -net_cost
                        losses += 1
                else:
                    pnl = Decimal("0")

                total_pnl += pnl
                pnl_list.append(float(pnl))

        # Calculate win rate
        total_resolved = wins + losses
        win_rate = wins / total_resolved if total_resolved > 0 else 0.0

        # Calculate timing metrics
        avg_time_to_resolution = self._calculate_avg_time_to_resolution(sorted_trades)
        pct_before_news = self._calculate_pre_news_percentage(sorted_trades)

        # Calculate Sharpe ratio (simplified)
        sharpe = 0.0
        if pnl_list and len(pnl_list) > 1:
            returns = np.array(pnl_list)
            if np.std(returns) > 0:
                sharpe = float(np.mean(returns) / np.std(returns))

        # Calculate profit factor
        gross_profit = sum(p for p in pnl_list if p > 0)
        gross_loss = abs(sum(p for p in pnl_list if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown(pnl_list)

        profile = AccountProfile(
            id=account_id,
            platform=platform,
            address=account_id,
            total_trades=len(sorted_trades),
            total_markets=len(unique_markets),
            total_volume=total_volume,
            total_pnl=total_pnl,
            realized_pnl=total_pnl,
            win_rate=win_rate,
            avg_return_per_trade=float(total_pnl) / len(sorted_trades) if sorted_trades else 0.0,
            sharpe_ratio=sharpe,
            avg_time_to_resolution=avg_time_to_resolution,
            pct_trades_before_news=pct_before_news,
            first_seen=sorted_trades[0].timestamp if sorted_trades else None,
            last_seen=sorted_trades[-1].timestamp if sorted_trades else None,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
        )

        self._profiles[account_key] = profile
        return profile

    def _calculate_avg_time_to_resolution(self, trades: list[Trade]) -> float:
        """Calculate average time between trade and market resolution."""
        times = []
        for trade in trades:
            market_key = f"{trade.platform.value}:{trade.market_id}"
            market = self._resolved_markets.get(market_key)
            if market and market.end_date:
                hours = (market.end_date - trade.timestamp).total_seconds() / 3600
                if hours > 0:
                    times.append(hours)
        return sum(times) / len(times) if times else 0.0

    def _calculate_pre_news_percentage(self, trades: list[Trade]) -> float:
        """Calculate percentage of trades that occurred before major price moves."""
        # This would need price history data to calculate properly
        # For now, return 0 as placeholder
        return 0.0

    def _calculate_max_drawdown(self, pnl_list: list[float]) -> float:
        """Calculate maximum drawdown from P&L sequence."""
        if not pnl_list:
            return 0.0

        cumulative = np.cumsum(pnl_list)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative

        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def detect_patterns(self, platform: Platform, account_id: str) -> list[TradingPattern]:
        """Detect trading patterns for an account."""
        patterns = []
        trades = self.get_trades(platform, account_id)

        if len(trades) < self.config.min_trades_for_analysis:
            return patterns

        # Pattern 1: Concentrated betting (large positions on single markets)
        concentrated = self._detect_concentrated_betting(trades)
        if concentrated:
            patterns.append(concentrated)

        # Pattern 2: Pre-resolution timing (trades close to resolution)
        timing = self._detect_timing_pattern(trades)
        if timing:
            patterns.append(timing)

        # Pattern 3: Low-probability winner (consistently wins long-shot bets)
        longshot = self._detect_longshot_winner(trades)
        if longshot:
            patterns.append(longshot)

        # Pattern 4: Category specialist (focuses on specific types of markets)
        specialist = self._detect_category_specialist(trades)
        if specialist:
            patterns.append(specialist)

        return patterns

    def _detect_concentrated_betting(self, trades: list[Trade]) -> Optional[TradingPattern]:
        """Detect if account makes concentrated bets."""
        by_market = defaultdict(Decimal)
        for trade in trades:
            by_market[trade.market_id] += trade.price * trade.size

        total = sum(by_market.values())
        if total == 0:
            return None

        # Check if any single market is > 50% of volume
        for market_id, volume in by_market.items():
            concentration = float(volume / total)
            if concentration > 0.5 and float(volume) > self.config.large_position_usd:
                return TradingPattern(
                    pattern_type="concentrated_betting",
                    confidence=concentration,
                    description=f"Account concentrates {concentration:.0%} of volume on single market",
                    evidence=[f"Market {market_id}: ${float(volume):,.2f}"],
                    affected_markets=[market_id],
                )

        return None

    def _detect_timing_pattern(self, trades: list[Trade]) -> Optional[TradingPattern]:
        """Detect trades suspiciously close to resolution."""
        suspicious_trades = []

        for trade in trades:
            market_key = f"{trade.platform.value}:{trade.market_id}"
            market = self._resolved_markets.get(market_key)

            if market and market.end_date:
                hours_to_resolution = (market.end_date - trade.timestamp).total_seconds() / 3600
                if 0 < hours_to_resolution < self.config.suspicious_hours_before_resolution:
                    suspicious_trades.append((trade, hours_to_resolution))

        if len(suspicious_trades) >= 3:
            pct = len(suspicious_trades) / len(trades)
            return TradingPattern(
                pattern_type="pre_resolution_timing",
                confidence=min(pct * 2, 1.0),  # Cap at 1.0
                description=f"{len(suspicious_trades)} trades made within {self.config.suspicious_hours_before_resolution}h of resolution",
                evidence=[f"Trade {t.id}: {h:.1f}h before resolution" for t, h in suspicious_trades[:5]],
                affected_markets=list(set(t.market_id for t, _ in suspicious_trades)),
            )

        return None

    def _detect_longshot_winner(self, trades: list[Trade]) -> Optional[TradingPattern]:
        """Detect accounts winning on low-probability bets."""
        longshot_wins = []

        for trade in trades:
            market_key = f"{trade.platform.value}:{trade.market_id}"
            market = self._resolved_markets.get(market_key)

            if market and market.outcome is not None:
                # Check if bet on low-probability side that won
                implied_prob = float(trade.implied_probability)
                bet_on_yes = trade.is_yes

                if implied_prob < self.config.low_probability_threshold:
                    # Bet on underdog
                    if (bet_on_yes and market.outcome) or (not bet_on_yes and not market.outcome):
                        longshot_wins.append((trade, implied_prob))

        if len(longshot_wins) >= 2:
            return TradingPattern(
                pattern_type="longshot_winner",
                confidence=len(longshot_wins) / 10,  # Rough confidence
                description=f"Won {len(longshot_wins)} bets on low-probability outcomes (<{self.config.low_probability_threshold:.0%})",
                evidence=[f"Market {t.market_id}: {p:.1%} probability" for t, p in longshot_wins[:5]],
                affected_markets=list(set(t.market_id for t, _ in longshot_wins)),
            )

        return None

    def _detect_category_specialist(self, trades: list[Trade]) -> Optional[TradingPattern]:
        """Detect if account specializes in specific categories."""
        # Would need market category data
        return None

    def get_profile(self, platform: Platform, account_id: str) -> Optional[AccountProfile]:
        """Get cached profile for an account."""
        return self._profiles.get(self._get_account_key(platform, account_id))

    def get_all_profiles(self) -> list[AccountProfile]:
        """Get all analyzed profiles."""
        return list(self._profiles.values())

    def rank_accounts_by_suspicion(self) -> list[tuple[AccountProfile, float]]:
        """Rank all accounts by suspicion score."""
        ranked = []

        for profile in self._profiles.values():
            score = self._calculate_suspicion_score(profile)
            profile.suspicion_score = score
            ranked.append((profile, score))

        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def _calculate_suspicion_score(self, profile: AccountProfile) -> float:
        """Calculate overall suspicion score for a profile."""
        score = 0.0
        reasons = []

        # Factor 1: Win rate (0-0.3)
        if profile.total_trades >= self.config.min_trades_for_analysis:
            if profile.win_rate >= self.config.very_suspicious_win_rate:
                score += 0.3
                reasons.append(f"Very high win rate: {profile.win_rate:.1%}")
            elif profile.win_rate >= self.config.suspicious_win_rate:
                score += 0.2
                reasons.append(f"High win rate: {profile.win_rate:.1%}")

        # Factor 2: Profit (0-0.3)
        profit = float(profile.total_pnl)
        if profit >= self.config.whale_profit_usd:
            score += 0.3
            reasons.append(f"Whale-level profit: ${profit:,.0f}")
        elif profit >= self.config.suspicious_profit_usd:
            score += 0.2
            reasons.append(f"Large profit: ${profit:,.0f}")

        # Factor 3: Timing (0-0.2)
        if profile.avg_time_to_resolution < self.config.suspicious_hours_before_resolution:
            score += 0.2
            reasons.append(f"Trades avg {profile.avg_time_to_resolution:.1f}h before resolution")

        # Factor 4: Profit factor (0-0.2)
        if profile.profit_factor > 3.0:
            score += 0.2
            reasons.append(f"High profit factor: {profile.profit_factor:.1f}")

        profile.suspicion_reasons = reasons
        return min(score, 1.0)
