"""
SQLite database for persistent storage of trades, accounts, and alerts.
"""

import json
import logging
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import aiosqlite

from ..config import get_config
from ..models import AccountProfile, Alert, Market, Platform, SuspiciousActivity, Trade

logger = logging.getLogger(__name__)


class Database:
    """SQLite database for storing trading data and analysis results."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_config().storage.database_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self) -> None:
        """Connect to database and initialize schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._create_schema()

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _create_schema(self) -> None:
        """Create database tables."""
        await self._conn.executescript("""
            -- Markets table
            CREATE TABLE IF NOT EXISTS markets (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                end_date TEXT,
                created_at TEXT,
                status TEXT,
                outcome INTEGER,
                volume TEXT,
                category TEXT,
                url TEXT,
                current_yes_price TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- Trades table
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                market_id TEXT NOT NULL,
                account_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                is_yes INTEGER NOT NULL,
                price TEXT NOT NULL,
                size TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                UNIQUE(platform, market_id, id)
            );

            -- Account profiles table
            CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                address TEXT,
                total_trades INTEGER DEFAULT 0,
                total_markets INTEGER DEFAULT 0,
                total_volume TEXT DEFAULT '0',
                total_pnl TEXT DEFAULT '0',
                win_rate REAL DEFAULT 0,
                suspicion_score REAL DEFAULT 0,
                suspicion_reasons TEXT,
                first_seen TEXT,
                last_seen TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(platform, id)
            );

            -- Suspicious activity table
            CREATE TABLE IF NOT EXISTS suspicious_activity (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                market_id TEXT NOT NULL,
                market_title TEXT,
                detection_type TEXT NOT NULL,
                severity REAL NOT NULL,
                description TEXT,
                detected_at TEXT NOT NULL,
                total_position_size TEXT,
                potential_profit TEXT
            );

            -- Alerts table
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                platform TEXT NOT NULL,
                market_id TEXT NOT NULL,
                market_title TEXT,
                alert_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                suspicious_accounts TEXT,
                total_suspicious_volume TEXT,
                current_price TEXT,
                recommended_position TEXT,
                confidence REAL,
                reasoning TEXT
            );

            -- Watched accounts (accounts we want to track)
            CREATE TABLE IF NOT EXISTS watched_accounts (
                account_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                reason TEXT,
                added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (account_id, platform)
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
            CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id);
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_accounts_suspicion ON accounts(suspicion_score DESC);
            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC);
        """)
        await self._conn.commit()

    # Market operations

    async def save_market(self, market: Market) -> None:
        """Save or update a market."""
        await self._conn.execute("""
            INSERT OR REPLACE INTO markets
            (id, platform, title, description, end_date, created_at, status,
             outcome, volume, category, url, current_yes_price, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market.id,
            market.platform.value,
            market.title,
            market.description,
            market.end_date.isoformat() if market.end_date else None,
            market.created_at.isoformat() if market.created_at else None,
            market.status.value,
            1 if market.outcome else 0 if market.outcome is False else None,
            str(market.volume),
            market.category,
            market.url,
            str(market.current_yes_price),
            datetime.now().isoformat(),
        ))
        await self._conn.commit()

    async def get_market(self, market_id: str) -> Optional[Market]:
        """Get a market by ID."""
        async with self._conn.execute(
            "SELECT * FROM markets WHERE id = ?", (market_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return self._row_to_market(row)
        return None

    def _row_to_market(self, row) -> Market:
        """Convert a database row to a Market object."""
        from ..models import MarketStatus

        return Market(
            id=row[0],
            platform=Platform(row[1]),
            title=row[2],
            description=row[3] or "",
            end_date=datetime.fromisoformat(row[4]) if row[4] else None,
            created_at=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
            status=MarketStatus(row[6]) if row[6] else MarketStatus.OPEN,
            outcome=True if row[7] == 1 else False if row[7] == 0 else None,
            volume=Decimal(row[8]) if row[8] else Decimal("0"),
            category=row[9] or "",
            url=row[10] or "",
            current_yes_price=Decimal(row[11]) if row[11] else Decimal("0.5"),
        )

    # Trade operations

    async def save_trades(self, trades: list[Trade]) -> int:
        """Save multiple trades. Returns count saved."""
        count = 0
        for trade in trades:
            try:
                await self._conn.execute("""
                    INSERT OR IGNORE INTO trades
                    (id, platform, market_id, account_id, direction, is_yes, price, size, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.id,
                    trade.platform.value,
                    trade.market_id,
                    trade.account_id,
                    trade.direction.value,
                    1 if trade.is_yes else 0,
                    str(trade.price),
                    str(trade.size),
                    trade.timestamp.isoformat(),
                ))
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save trade {trade.id}: {e}")
        await self._conn.commit()
        return count

    async def get_trades(
        self,
        market_id: Optional[str] = None,
        account_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[Trade]:
        """Get trades with optional filters."""
        from ..models import TradeDirection

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if market_id:
            query += " AND market_id = ?"
            params.append(market_id)
        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        trades = []
        async with self._conn.execute(query, params) as cursor:
            async for row in cursor:
                trades.append(Trade(
                    id=row[0],
                    platform=Platform(row[1]),
                    market_id=row[2],
                    account_id=row[3],
                    direction=TradeDirection(row[4]),
                    is_yes=row[5] == 1,
                    price=Decimal(row[6]),
                    size=Decimal(row[7]),
                    timestamp=datetime.fromisoformat(row[8]),
                ))

        return trades

    # Account operations

    async def save_account(self, profile: AccountProfile) -> None:
        """Save or update an account profile."""
        await self._conn.execute("""
            INSERT OR REPLACE INTO accounts
            (id, platform, address, total_trades, total_markets, total_volume,
             total_pnl, win_rate, suspicion_score, suspicion_reasons,
             first_seen, last_seen, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile.id,
            profile.platform.value,
            profile.address,
            profile.total_trades,
            profile.total_markets,
            str(profile.total_volume),
            str(profile.total_pnl),
            profile.win_rate,
            profile.suspicion_score,
            json.dumps(profile.suspicion_reasons),
            profile.first_seen.isoformat() if profile.first_seen else None,
            profile.last_seen.isoformat() if profile.last_seen else None,
            datetime.now().isoformat(),
        ))
        await self._conn.commit()

    async def get_suspicious_accounts(
        self,
        min_score: float = 0.5,
        limit: int = 100,
    ) -> list[AccountProfile]:
        """Get accounts with high suspicion scores."""
        profiles = []
        async with self._conn.execute("""
            SELECT * FROM accounts
            WHERE suspicion_score >= ?
            ORDER BY suspicion_score DESC
            LIMIT ?
        """, (min_score, limit)) as cursor:
            async for row in cursor:
                profiles.append(AccountProfile(
                    id=row[0],
                    platform=Platform(row[1]),
                    address=row[2] or row[0],
                    total_trades=row[3] or 0,
                    total_markets=row[4] or 0,
                    total_volume=Decimal(row[5]) if row[5] else Decimal("0"),
                    total_pnl=Decimal(row[6]) if row[6] else Decimal("0"),
                    win_rate=row[7] or 0.0,
                    suspicion_score=row[8] or 0.0,
                    suspicion_reasons=json.loads(row[9]) if row[9] else [],
                    first_seen=datetime.fromisoformat(row[10]) if row[10] else None,
                    last_seen=datetime.fromisoformat(row[11]) if row[11] else None,
                ))

        return profiles

    async def add_watched_account(
        self,
        platform: Platform,
        account_id: str,
        reason: str = "",
    ) -> None:
        """Add an account to the watch list."""
        await self._conn.execute("""
            INSERT OR REPLACE INTO watched_accounts (account_id, platform, reason, added_at)
            VALUES (?, ?, ?, ?)
        """, (account_id, platform.value, reason, datetime.now().isoformat()))
        await self._conn.commit()

    async def get_watched_accounts(self) -> list[tuple[str, Platform, str]]:
        """Get all watched accounts."""
        accounts = []
        async with self._conn.execute(
            "SELECT account_id, platform, reason FROM watched_accounts"
        ) as cursor:
            async for row in cursor:
                accounts.append((row[0], Platform(row[1]), row[2] or ""))
        return accounts

    # Alert operations

    async def save_alert(self, alert: Alert) -> None:
        """Save an alert."""
        await self._conn.execute("""
            INSERT OR REPLACE INTO alerts
            (id, timestamp, platform, market_id, market_title, alert_type,
             priority, suspicious_accounts, total_suspicious_volume,
             current_price, recommended_position, confidence, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.id,
            alert.timestamp.isoformat(),
            alert.platform.value,
            alert.market_id,
            alert.market_title,
            alert.alert_type,
            alert.priority,
            json.dumps(alert.suspicious_accounts),
            str(alert.total_suspicious_volume),
            str(alert.current_price),
            alert.recommended_position,
            alert.confidence,
            alert.reasoning,
        ))
        await self._conn.commit()

    async def get_recent_alerts(
        self,
        hours: int = 24,
        priority: Optional[str] = None,
        limit: int = 100,
    ) -> list[Alert]:
        """Get recent alerts."""
        cutoff = datetime.now().replace(
            hour=datetime.now().hour - hours if datetime.now().hour >= hours else 0
        )

        query = "SELECT * FROM alerts WHERE timestamp >= ?"
        params = [cutoff.isoformat()]

        if priority:
            query += " AND priority = ?"
            params.append(priority)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        alerts = []
        async with self._conn.execute(query, params) as cursor:
            async for row in cursor:
                alerts.append(Alert(
                    id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    platform=Platform(row[2]),
                    market_id=row[3],
                    market_title=row[4] or "",
                    alert_type=row[5],
                    priority=row[6],
                    suspicious_accounts=json.loads(row[7]) if row[7] else [],
                    total_suspicious_volume=Decimal(row[8]) if row[8] else Decimal("0"),
                    current_price=Decimal(row[9]) if row[9] else Decimal("0"),
                    recommended_position=row[10] or "",
                    confidence=row[11] or 0.0,
                    reasoning=row[12] or "",
                ))

        return alerts

    # Statistics

    async def get_stats(self) -> dict:
        """Get database statistics."""
        stats = {}

        async with self._conn.execute("SELECT COUNT(*) FROM markets") as cursor:
            stats["markets"] = (await cursor.fetchone())[0]

        async with self._conn.execute("SELECT COUNT(*) FROM trades") as cursor:
            stats["trades"] = (await cursor.fetchone())[0]

        async with self._conn.execute("SELECT COUNT(*) FROM accounts") as cursor:
            stats["accounts"] = (await cursor.fetchone())[0]

        async with self._conn.execute("SELECT COUNT(*) FROM alerts") as cursor:
            stats["alerts"] = (await cursor.fetchone())[0]

        async with self._conn.execute(
            "SELECT COUNT(*) FROM accounts WHERE suspicion_score >= 0.5"
        ) as cursor:
            stats["suspicious_accounts"] = (await cursor.fetchone())[0]

        return stats
