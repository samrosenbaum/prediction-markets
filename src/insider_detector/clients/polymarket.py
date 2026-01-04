"""
Polymarket API client.

Polymarket uses multiple APIs:
- CLOB API: For order book and trading
- Gamma API: For market metadata and leaderboards
- The Graph: For historical on-chain data
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import AsyncGenerator, Optional

import httpx

from ..config import PolymarketConfig, get_config
from ..models import (
    AccountProfile,
    Market,
    MarketStatus,
    Platform,
    Position,
    Trade,
    TradeDirection,
)
from .base import BaseClient

logger = logging.getLogger(__name__)


class PolymarketClient(BaseClient):
    """Client for interacting with Polymarket APIs."""

    def __init__(self, config: Optional[PolymarketConfig] = None):
        self.config = config or get_config().polymarket
        super().__init__(
            platform=Platform.POLYMARKET,
            base_url=self.config.clob_api_url,
            requests_per_second=self.config.requests_per_second,
        )
        self._gamma_client: Optional[httpx.AsyncClient] = None

    async def connect(self) -> None:
        """Initialize HTTP clients."""
        await super().connect()
        self._gamma_client = httpx.AsyncClient(
            base_url=self.config.gamma_api_url,
            timeout=30.0,
            headers={"User-Agent": "InsiderTradingDetector/0.1.0"},
        )

    async def close(self) -> None:
        """Close HTTP clients."""
        await super().close()
        if self._gamma_client:
            await self._gamma_client.aclose()
            self._gamma_client = None

    async def _gamma_get(self, path: str, params: Optional[dict] = None) -> dict:
        """Make a request to the Gamma API."""
        if not self._gamma_client:
            raise RuntimeError("Client not connected")

        await self.rate_limiter.acquire()
        response = await self._gamma_client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    def _parse_market(self, data: dict) -> Market:
        """Parse market data from API response."""
        status_map = {
            "active": MarketStatus.OPEN,
            "closed": MarketStatus.CLOSED,
            "resolved": MarketStatus.RESOLVED,
        }

        # Handle outcome - resolved markets have a result
        outcome = None
        if data.get("resolved"):
            # Check if YES outcome won
            outcome = data.get("outcome") == "Yes" or data.get("winner") == "Yes"

        # Parse end date
        end_date = None
        if data.get("endDate") or data.get("end_date_iso"):
            try:
                end_str = data.get("endDate") or data.get("end_date_iso")
                if end_str:
                    end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Parse created date
        created_at = datetime.now()
        if data.get("createdAt") or data.get("created_at"):
            try:
                created_str = data.get("createdAt") or data.get("created_at")
                if created_str:
                    created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return Market(
            id=data.get("condition_id") or data.get("id") or data.get("clob_token_ids", [""])[0],
            platform=Platform.POLYMARKET,
            title=data.get("question") or data.get("title", ""),
            description=data.get("description", ""),
            end_date=end_date,
            created_at=created_at,
            status=status_map.get(data.get("active", "active"), MarketStatus.OPEN),
            outcome=outcome,
            volume=Decimal(str(data.get("volume", 0) or data.get("volumeNum", 0) or 0)),
            liquidity=Decimal(str(data.get("liquidity", 0) or data.get("liquidityNum", 0) or 0)),
            category=data.get("category", "") or data.get("groupItemTitle", ""),
            tags=data.get("tags", []) or [],
            url=f"https://polymarket.com/event/{data.get('slug', data.get('id', ''))}",
            current_yes_price=Decimal(str(data.get("outcomePrices", [0.5])[0] if isinstance(data.get("outcomePrices"), list) else 0.5)),
            current_no_price=Decimal(str(data.get("outcomePrices", [0.5, 0.5])[1] if isinstance(data.get("outcomePrices"), list) and len(data.get("outcomePrices", [])) > 1 else 0.5)),
        )

    def _parse_trade(self, data: dict, market_id: str = "") -> Trade:
        """Parse trade data from API response."""
        # Determine direction
        side = data.get("side", "").lower()
        direction = TradeDirection.BUY if side == "buy" else TradeDirection.SELL

        # Determine if YES or NO
        is_yes = data.get("outcome", "").lower() == "yes" or data.get("asset_id", "").endswith("yes")

        # Parse timestamp
        timestamp = datetime.now()
        if data.get("timestamp") or data.get("created_at"):
            try:
                ts = data.get("timestamp") or data.get("created_at")
                if isinstance(ts, (int, float)):
                    timestamp = datetime.fromtimestamp(ts)
                else:
                    timestamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return Trade(
            id=data.get("id") or data.get("trade_id") or str(data.get("timestamp", "")),
            platform=Platform.POLYMARKET,
            market_id=market_id or data.get("market", "") or data.get("condition_id", ""),
            account_id=data.get("maker", "") or data.get("taker", "") or data.get("owner", ""),
            direction=direction,
            is_yes=is_yes,
            price=Decimal(str(data.get("price", 0))),
            size=Decimal(str(data.get("size", 0) or data.get("amount", 0))),
            timestamp=timestamp,
        )

    async def get_markets(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        """Fetch list of markets from Gamma API."""
        params = {
            "limit": limit,
            "offset": offset,
        }
        if status:
            params["active"] = status == "active"
            params["closed"] = status == "closed"

        try:
            data = await self._gamma_get("/markets", params=params)
            markets = []
            for item in data if isinstance(data, list) else data.get("data", []):
                try:
                    markets.append(self._parse_market(item))
                except Exception as e:
                    logger.warning(f"Failed to parse market: {e}")
            return markets
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def get_market(self, market_id: str) -> Market:
        """Fetch a single market by ID."""
        data = await self._gamma_get(f"/markets/{market_id}")
        return self._parse_market(data)

    async def get_trades(
        self,
        market_id: Optional[str] = None,
        account_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Trade]:
        """Fetch trades from CLOB API."""
        params = {"limit": limit}

        if market_id:
            params["market"] = market_id
        if account_id:
            params["maker"] = account_id

        try:
            # Use CLOB API for trades
            data = await self.get("/trades", params=params)
            trades = []
            for item in data if isinstance(data, list) else data.get("data", []):
                try:
                    trade = self._parse_trade(item, market_id or "")
                    # Filter by time if specified
                    if start_time and trade.timestamp < start_time:
                        continue
                    if end_time and trade.timestamp > end_time:
                        continue
                    trades.append(trade)
                except Exception as e:
                    logger.warning(f"Failed to parse trade: {e}")
            return trades
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []

    async def get_market_trades(
        self,
        token_id: str,
        limit: int = 500,
    ) -> list[Trade]:
        """Fetch trades for a specific market token."""
        try:
            params = {"limit": limit}
            data = await self.get(f"/trades/{token_id}", params=params)
            trades = []
            items = data if isinstance(data, list) else data.get("data", data.get("trades", []))
            for item in items:
                try:
                    trades.append(self._parse_trade(item, token_id))
                except Exception as e:
                    logger.warning(f"Failed to parse trade: {e}")
            return trades
        except Exception as e:
            logger.error(f"Failed to fetch market trades: {e}")
            return []

    async def stream_trades(
        self,
        market_ids: Optional[list[str]] = None,
    ) -> AsyncGenerator[Trade, None]:
        """Stream live trades (polling-based for now)."""
        import asyncio

        last_seen: dict[str, str] = {}

        while True:
            try:
                # Fetch recent trades
                if market_ids:
                    for market_id in market_ids:
                        trades = await self.get_trades(market_id=market_id, limit=50)
                        for trade in trades:
                            trade_key = f"{trade.market_id}:{trade.id}"
                            if trade_key not in last_seen:
                                last_seen[trade_key] = trade.id
                                yield trade
                else:
                    trades = await self.get_trades(limit=50)
                    for trade in trades:
                        trade_key = f"{trade.market_id}:{trade.id}"
                        if trade_key not in last_seen:
                            last_seen[trade_key] = trade.id
                            yield trade

                # Cleanup old entries to prevent memory leak
                if len(last_seen) > 10000:
                    # Keep only the last 5000
                    items = list(last_seen.items())[-5000:]
                    last_seen = dict(items)

            except Exception as e:
                logger.error(f"Error streaming trades: {e}")

            await asyncio.sleep(5)  # Poll every 5 seconds

    async def get_account_positions(self, account_id: str) -> list[Position]:
        """Get all positions for an account."""
        try:
            data = await self._gamma_get(f"/positions", params={"user": account_id})
            positions = []
            items = data if isinstance(data, list) else data.get("data", data.get("positions", []))

            for item in items:
                try:
                    pos = Position(
                        account_id=account_id,
                        platform=Platform.POLYMARKET,
                        market_id=item.get("conditionId", "") or item.get("market", ""),
                        yes_shares=Decimal(str(item.get("yesShares", 0) or item.get("size", 0))),
                        no_shares=Decimal(str(item.get("noShares", 0))),
                        avg_yes_price=Decimal(str(item.get("avgPrice", 0) or item.get("averagePrice", 0))),
                        total_invested=Decimal(str(item.get("invested", 0) or item.get("cost", 0))),
                        realized_pnl=Decimal(str(item.get("realizedPnl", 0))),
                        unrealized_pnl=Decimal(str(item.get("unrealizedPnl", 0) or item.get("currentValue", 0))),
                    )
                    positions.append(pos)
                except Exception as e:
                    logger.warning(f"Failed to parse position: {e}")

            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions for {account_id}: {e}")
            return []

    async def get_account_profile(self, account_id: str) -> AccountProfile:
        """Get profile/statistics for an account."""
        try:
            # Fetch from leaderboard or profile API
            data = await self._gamma_get(f"/profiles/{account_id}")

            profile = AccountProfile(
                id=account_id,
                platform=Platform.POLYMARKET,
                address=account_id,
                total_trades=data.get("tradesCount", 0) or data.get("totalTrades", 0),
                total_markets=data.get("marketsTraded", 0) or data.get("totalMarkets", 0),
                total_volume=Decimal(str(data.get("volume", 0) or data.get("totalVolume", 0))),
                total_pnl=Decimal(str(data.get("pnl", 0) or data.get("profit", 0))),
                realized_pnl=Decimal(str(data.get("realizedPnl", 0))),
                win_rate=float(data.get("winRate", 0) or 0),
            )

            # Get positions
            profile.positions = await self.get_account_positions(account_id)

            return profile
        except Exception as e:
            logger.error(f"Failed to fetch profile for {account_id}: {e}")
            # Return basic profile
            return AccountProfile(
                id=account_id,
                platform=Platform.POLYMARKET,
                address=account_id,
            )

    async def get_top_traders(
        self,
        market_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[AccountProfile]:
        """Get top traders by volume or profit."""
        try:
            params = {"limit": limit}
            if market_id:
                params["market"] = market_id

            data = await self._gamma_get("/leaderboard", params=params)
            profiles = []

            items = data if isinstance(data, list) else data.get("data", data.get("leaderboard", []))
            for item in items:
                try:
                    address = item.get("address", "") or item.get("user", "") or item.get("id", "")
                    profile = AccountProfile(
                        id=address,
                        platform=Platform.POLYMARKET,
                        address=address,
                        total_trades=item.get("tradesCount", 0),
                        total_volume=Decimal(str(item.get("volume", 0))),
                        total_pnl=Decimal(str(item.get("pnl", 0) or item.get("profit", 0))),
                        realized_pnl=Decimal(str(item.get("realizedPnl", 0))),
                        win_rate=float(item.get("winRate", 0) or 0),
                    )
                    profiles.append(profile)
                except Exception as e:
                    logger.warning(f"Failed to parse leaderboard entry: {e}")

            return profiles
        except Exception as e:
            logger.error(f"Failed to fetch top traders: {e}")
            return []

    async def get_market_activity(
        self,
        market_id: str,
        hours: int = 24,
    ) -> dict:
        """Get recent trading activity for a market."""
        try:
            data = await self._gamma_get(f"/markets/{market_id}/activity", params={"hours": hours})
            return {
                "volume_24h": Decimal(str(data.get("volume24h", 0))),
                "trades_24h": data.get("trades24h", 0),
                "unique_traders": data.get("uniqueTraders", 0),
                "price_change": Decimal(str(data.get("priceChange", 0))),
                "top_buyers": data.get("topBuyers", []),
                "top_sellers": data.get("topSellers", []),
            }
        except Exception as e:
            logger.error(f"Failed to fetch market activity: {e}")
            return {}

    async def search_markets(self, query: str, limit: int = 20) -> list[Market]:
        """Search for markets by keyword."""
        try:
            data = await self._gamma_get("/markets", params={"query": query, "limit": limit})
            return [self._parse_market(m) for m in (data if isinstance(data, list) else data.get("data", []))]
        except Exception as e:
            logger.error(f"Failed to search markets: {e}")
            return []
