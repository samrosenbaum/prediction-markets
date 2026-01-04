"""
Configuration management for the insider trading detector.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class PolymarketConfig:
    """Configuration for Polymarket API."""
    # Polymarket uses The Graph for historical data and CLOB API for live data
    clob_api_url: str = "https://clob.polymarket.com"
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    subgraph_url: str = "https://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic"

    # Rate limiting
    requests_per_second: int = 5

    # Chain config (Polygon)
    chain_id: int = 137
    ctf_exchange_address: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    neg_risk_ctf_exchange: str = "0xC5d563A36AE78145C45a50134d48A1215220f80a"


@dataclass
class KalshiConfig:
    """Configuration for Kalshi API."""
    api_url: str = "https://trading-api.kalshi.com/trade-api/v2"
    demo_api_url: str = "https://demo-api.kalshi.co/trade-api/v2"

    # Authentication (required for some endpoints)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    # Rate limiting
    requests_per_second: int = 10

    # Use demo mode for testing
    use_demo: bool = False


@dataclass
class DetectionConfig:
    """Configuration for insider trading detection algorithms."""

    # Timing thresholds
    suspicious_hours_before_resolution: float = 48.0  # Trades this close to resolution are flagged
    news_window_hours: float = 24.0  # Window for detecting pre-news trades

    # Win rate thresholds
    min_trades_for_analysis: int = 5  # Minimum trades to analyze an account
    suspicious_win_rate: float = 0.75  # Win rate above this is flagged
    very_suspicious_win_rate: float = 0.85

    # Position size thresholds
    large_position_usd: float = 5000.0  # Positions above this are considered large
    whale_position_usd: float = 50000.0

    # Profit thresholds
    suspicious_profit_usd: float = 10000.0
    whale_profit_usd: float = 100000.0

    # Probability thresholds
    low_probability_threshold: float = 0.20  # Bets on <20% outcomes that win
    extreme_probability_threshold: float = 0.10

    # Clustering
    min_correlated_accounts: int = 3  # Minimum accounts for cluster detection
    correlation_threshold: float = 0.8

    # Alert thresholds
    min_severity_for_alert: float = 0.5
    min_confidence_for_recommendation: float = 0.6


@dataclass
class StorageConfig:
    """Configuration for data storage."""
    database_path: Path = field(default_factory=lambda: Path("data/insider_detector.db"))
    cache_ttl_seconds: int = 300  # 5 minutes
    max_cached_markets: int = 1000
    max_cached_accounts: int = 10000


@dataclass
class Config:
    """Main configuration container."""
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Global settings
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        load_dotenv()

        config = cls()

        # Kalshi credentials
        config.kalshi.api_key = os.getenv("KALSHI_API_KEY")
        config.kalshi.api_secret = os.getenv("KALSHI_API_SECRET")
        config.kalshi.use_demo = os.getenv("KALSHI_USE_DEMO", "false").lower() == "true"

        # Debug mode
        config.debug = os.getenv("DEBUG", "false").lower() == "true"
        config.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Storage
        db_path = os.getenv("DATABASE_PATH")
        if db_path:
            config.storage.database_path = Path(db_path)

        return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
