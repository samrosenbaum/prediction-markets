"""
Base client class for API interactions.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import AsyncGenerator, Optional

import httpx

from ..models import AccountProfile, Market, Platform, Position, Trade

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_second: int):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until we can make another request."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            self.last_request_time = asyncio.get_event_loop().time()


class BaseClient(ABC):
    """Base class for prediction market API clients."""

    def __init__(
        self,
        platform: Platform,
        base_url: str,
        requests_per_second: int = 5,
    ):
        self.platform = platform
        self.base_url = base_url
        self.rate_limiter = RateLimiter(requests_per_second)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "BaseClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers={"User-Agent": "InsiderTradingDetector/0.1.0"},
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> dict:
        """Make a rate-limited HTTP request."""
        if not self._client:
            raise RuntimeError("Client not connected. Use 'async with' or call connect()")

        await self.rate_limiter.acquire()

        try:
            response = await self._client.request(
                method=method,
                url=path,
                params=params,
                json=json,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

    async def get(self, path: str, params: Optional[dict] = None) -> dict:
        """Make a GET request."""
        return await self._request("GET", path, params=params)

    async def post(self, path: str, json: Optional[dict] = None) -> dict:
        """Make a POST request."""
        return await self._request("POST", path, json=json)

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    async def get_markets(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        """Fetch list of markets."""
        pass

    @abstractmethod
    async def get_market(self, market_id: str) -> Market:
        """Fetch a single market by ID."""
        pass

    @abstractmethod
    async def get_trades(
        self,
        market_id: Optional[str] = None,
        account_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Trade]:
        """Fetch trades, optionally filtered by market and/or account."""
        pass

    @abstractmethod
    async def stream_trades(
        self,
        market_ids: Optional[list[str]] = None,
    ) -> AsyncGenerator[Trade, None]:
        """Stream live trades."""
        pass

    @abstractmethod
    async def get_account_positions(self, account_id: str) -> list[Position]:
        """Get all positions for an account."""
        pass

    @abstractmethod
    async def get_account_profile(self, account_id: str) -> AccountProfile:
        """Get profile/statistics for an account."""
        pass

    @abstractmethod
    async def get_top_traders(
        self,
        market_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[AccountProfile]:
        """Get top traders by volume or profit."""
        pass
