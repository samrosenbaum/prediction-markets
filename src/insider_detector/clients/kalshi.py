"""
Kalshi API client.

Kalshi is a CFTC-regulated prediction market with a REST API.
Some endpoints require authentication.
"""

import hashlib
import hmac
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import AsyncGenerator, Optional

from ..config import KalshiConfig, get_config
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


class KalshiClient(BaseClient):
    """Client for interacting with Kalshi API."""

    def __init__(self, config: Optional[KalshiConfig] = None):
        self.config = config or get_config().kalshi
        base_url = self.config.demo_api_url if self.config.use_demo else self.config.api_url
        super().__init__(
            platform=Platform.KALSHI,
            base_url=base_url,
            requests_per_second=self.config.requests_per_second,
        )
        self._auth_token: Optional[str] = None

    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC signature for authenticated requests."""
        if not self.config.api_secret:
            return ""

        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.config.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature

    async def _auth_request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> dict:
        """Make an authenticated request."""
        if not self._client:
            raise RuntimeError("Client not connected")

        await self.rate_limiter.acquire()

        timestamp = str(int(time.time() * 1000))
        body = ""
        if json:
            import json as json_lib
            body = json_lib.dumps(json)

        headers = {
            "Content-Type": "application/json",
        }

        if self.config.api_key:
            signature = self._generate_signature(timestamp, method, path, body)
            headers.update({
                "KALSHI-ACCESS-KEY": self.config.api_key,
                "KALSHI-ACCESS-SIGNATURE": signature,
                "KALSHI-ACCESS-TIMESTAMP": timestamp,
            })

        response = await self._client.request(
            method=method,
            url=path,
            params=params,
            json=json,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    def _parse_market(self, data: dict) -> Market:
        """Parse market data from API response."""
        status_map = {
            "active": MarketStatus.OPEN,
            "closed": MarketStatus.CLOSED,
            "settled": MarketStatus.RESOLVED,
            "finalized": MarketStatus.RESOLVED,
        }

        # Parse outcome
        outcome = None
        if data.get("result"):
            outcome = data["result"].lower() == "yes"

        # Parse dates
        end_date = None
        if data.get("close_time") or data.get("expiration_time"):
            try:
                end_str = data.get("close_time") or data.get("expiration_time")
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        created_at = datetime.now()
        if data.get("open_time"):
            try:
                created_at = datetime.fromisoformat(data["open_time"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Get current price (Kalshi uses cents, 0-100)
        yes_price = Decimal(str(data.get("yes_bid", 50))) / 100
        no_price = Decimal("1") - yes_price

        return Market(
            id=data.get("ticker", "") or data.get("market_ticker", ""),
            platform=Platform.KALSHI,
            title=data.get("title", "") or data.get("subtitle", ""),
            description=data.get("rules_primary", "") or data.get("description", ""),
            end_date=end_date,
            created_at=created_at,
            status=status_map.get(data.get("status", "active"), MarketStatus.OPEN),
            outcome=outcome,
            volume=Decimal(str(data.get("volume", 0) or data.get("dollar_volume", 0))),
            liquidity=Decimal(str(data.get("open_interest", 0))),
            category=data.get("category", "") or data.get("series_ticker", ""),
            tags=data.get("tags", []) or [],
            url=f"https://kalshi.com/markets/{data.get('ticker', '')}",
            current_yes_price=yes_price,
            current_no_price=no_price,
        )

    def _parse_trade(self, data: dict, market_id: str = "") -> Trade:
        """Parse trade data from API response."""
        # Kalshi uses "yes"/"no" for side
        is_yes = data.get("side", "").lower() == "yes"

        # Taker side determines direction
        taker_side = data.get("taker_side", "")
        if taker_side:
            is_yes = taker_side.lower() == "yes"

        # Parse timestamp
        timestamp = datetime.now()
        if data.get("created_time") or data.get("ts"):
            try:
                ts = data.get("created_time") or data.get("ts")
                if isinstance(ts, (int, float)):
                    timestamp = datetime.fromtimestamp(ts / 1000)  # Kalshi uses milliseconds
                else:
                    timestamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Price in Kalshi is 0-100 cents
        price = Decimal(str(data.get("yes_price", 50) or data.get("price", 50))) / 100

        return Trade(
            id=data.get("trade_id", "") or str(data.get("id", "")),
            platform=Platform.KALSHI,
            market_id=market_id or data.get("ticker", "") or data.get("market_ticker", ""),
            account_id=data.get("user_id", "") or data.get("member_id", "") or "anonymous",
            direction=TradeDirection.BUY,  # Kalshi public trades don't show buy/sell
            is_yes=is_yes,
            price=price,
            size=Decimal(str(data.get("count", 0) or data.get("contracts", 0))),  # Number of contracts
            timestamp=timestamp,
        )

    async def get_markets(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        """Fetch list of markets."""
        params = {
            "limit": limit,
            "cursor": offset,
        }
        if status:
            params["status"] = status

        try:
            data = await self.get("/markets", params=params)
            markets = []
            for item in data.get("markets", []):
                try:
                    markets.append(self._parse_market(item))
                except Exception as e:
                    logger.warning(f"Failed to parse market: {e}")
            return markets
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def get_market(self, market_id: str) -> Market:
        """Fetch a single market by ticker."""
        data = await self.get(f"/markets/{market_id}")
        return self._parse_market(data.get("market", data))

    async def get_event_markets(self, event_ticker: str) -> list[Market]:
        """Get all markets for an event."""
        try:
            data = await self.get(f"/events/{event_ticker}/markets")
            return [self._parse_market(m) for m in data.get("markets", [])]
        except Exception as e:
            logger.error(f"Failed to fetch event markets: {e}")
            return []

    async def get_trades(
        self,
        market_id: Optional[str] = None,
        account_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Trade]:
        """Fetch trades for a market."""
        if not market_id:
            # Kalshi requires a market ticker for trades
            logger.warning("Kalshi requires market_id to fetch trades")
            return []

        params = {"limit": limit}
        if start_time:
            params["min_ts"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["max_ts"] = int(end_time.timestamp() * 1000)

        try:
            data = await self.get(f"/markets/{market_id}/trades", params=params)
            trades = []
            for item in data.get("trades", []):
                try:
                    trade = self._parse_trade(item, market_id)
                    trades.append(trade)
                except Exception as e:
                    logger.warning(f"Failed to parse trade: {e}")
            return trades
        except Exception as e:
            logger.error(f"Failed to fetch trades: {e}")
            return []

    async def get_market_history(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict]:
        """Get price history for a market."""
        params = {}
        if start_time:
            params["min_ts"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["max_ts"] = int(end_time.timestamp() * 1000)

        try:
            data = await self.get(f"/markets/{market_id}/history", params=params)
            return data.get("history", [])
        except Exception as e:
            logger.error(f"Failed to fetch market history: {e}")
            return []

    async def stream_trades(
        self,
        market_ids: Optional[list[str]] = None,
    ) -> AsyncGenerator[Trade, None]:
        """Stream live trades (polling-based)."""
        import asyncio

        if not market_ids:
            # Get active markets
            markets = await self.get_markets(status="active", limit=50)
            market_ids = [m.id for m in markets]

        last_seen: dict[str, str] = {}

        while True:
            try:
                for market_id in market_ids:
                    trades = await self.get_trades(market_id=market_id, limit=20)
                    for trade in trades:
                        trade_key = f"{trade.market_id}:{trade.id}"
                        if trade_key not in last_seen:
                            last_seen[trade_key] = trade.id
                            yield trade

                # Cleanup
                if len(last_seen) > 10000:
                    items = list(last_seen.items())[-5000:]
                    last_seen = dict(items)

            except Exception as e:
                logger.error(f"Error streaming trades: {e}")

            await asyncio.sleep(10)  # Poll every 10 seconds

    async def get_account_positions(self, account_id: str) -> list[Position]:
        """Get positions for an account (requires auth)."""
        if not self.config.api_key:
            logger.warning("Authentication required to fetch positions")
            return []

        try:
            data = await self._auth_request("GET", "/portfolio/positions")
            positions = []
            for item in data.get("positions", data.get("market_positions", [])):
                try:
                    pos = Position(
                        account_id=account_id,
                        platform=Platform.KALSHI,
                        market_id=item.get("ticker", "") or item.get("market_ticker", ""),
                        yes_shares=Decimal(str(item.get("position", 0))) if item.get("position", 0) > 0 else Decimal("0"),
                        no_shares=Decimal(str(abs(item.get("position", 0)))) if item.get("position", 0) < 0 else Decimal("0"),
                        total_invested=Decimal(str(item.get("total_cost", 0))),
                        realized_pnl=Decimal(str(item.get("realized_pnl", 0))),
                    )
                    positions.append(pos)
                except Exception as e:
                    logger.warning(f"Failed to parse position: {e}")
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    async def get_account_profile(self, account_id: str) -> AccountProfile:
        """Get account profile (requires auth for own account)."""
        # Kalshi doesn't expose public user profiles
        return AccountProfile(
            id=account_id,
            platform=Platform.KALSHI,
            address=account_id,
        )

    async def get_top_traders(
        self,
        market_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[AccountProfile]:
        """Get top traders - Kalshi doesn't have public leaderboard."""
        logger.info("Kalshi doesn't provide public leaderboard data")
        return []

    async def get_orderbook(self, market_id: str) -> dict:
        """Get current orderbook for a market."""
        try:
            data = await self.get(f"/markets/{market_id}/orderbook")
            return {
                "yes_bids": [(Decimal(str(o.get("price", 0))) / 100, o.get("quantity", 0))
                            for o in data.get("orderbook", {}).get("yes", [])],
                "no_bids": [(Decimal(str(o.get("price", 0))) / 100, o.get("quantity", 0))
                           for o in data.get("orderbook", {}).get("no", [])],
            }
        except Exception as e:
            logger.error(f"Failed to fetch orderbook: {e}")
            return {"yes_bids": [], "no_bids": []}

    async def search_markets(self, query: str, limit: int = 20) -> list[Market]:
        """Search for markets by keyword."""
        try:
            # Kalshi uses series/events for organization
            data = await self.get("/markets", params={"status": "active", "limit": limit})
            markets = []
            for m in data.get("markets", []):
                title = (m.get("title", "") + " " + m.get("subtitle", "")).lower()
                if query.lower() in title:
                    markets.append(self._parse_market(m))
            return markets
        except Exception as e:
            logger.error(f"Failed to search markets: {e}")
            return []

    async def get_events(self, status: str = "active", limit: int = 50) -> list[dict]:
        """Get list of events (groups of related markets)."""
        try:
            data = await self.get("/events", params={"status": status, "limit": limit})
            return data.get("events", [])
        except Exception as e:
            logger.error(f"Failed to fetch events: {e}")
            return []
