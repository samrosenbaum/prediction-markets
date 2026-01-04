# Insider Trading Detector for Prediction Markets

A tool to detect potential insider trading activity on prediction markets like Polymarket and Kalshi. Identifies suspicious trading patterns, tracks "smart money" accounts, and generates alerts for potential trading opportunities.

## Features

- **Multi-Platform Support**: Monitor both Polymarket and Kalshi
- **Account Profiling**: Track win rates, P&L, and trading patterns
- **Insider Detection**: Identify accounts with suspiciously high performance
- **Pattern Recognition**:
  - Pre-announcement trading surges
  - Coordinated trading across accounts
  - Whale accumulation patterns
  - Low-probability winners
- **Real-time Monitoring**: Stream live trades and get instant alerts
- **Trading Signals**: Get recommendations based on "smart money" activity

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/prediction-markets.git
cd prediction-markets

# Install with pip
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### Scan Markets for Suspicious Activity

```bash
# Scan Polymarket for suspicious accounts
insider-detector scan --platform polymarket --limit 50

# Scan both platforms
insider-detector scan --platform polymarket,kalshi --limit 100

# Only show highly suspicious accounts
insider-detector scan --min-suspicion 0.7
```

### Monitor Live Trades

```bash
# Monitor Polymarket in real-time
insider-detector monitor --platform polymarket

# Watch specific markets
insider-detector monitor --markets MARKET_ID_1,MARKET_ID_2
```

### Analyze Specific Accounts

```bash
# Analyze a Polymarket account
insider-detector account 0x1234567890abcdef... --platform polymarket

# Analyze a Kalshi account
insider-detector account user123 --platform kalshi
```

### Find Trading Opportunities

```bash
# Get current trading opportunities
insider-detector opportunities --min-confidence 0.6

# View recent alerts
insider-detector alerts --hours 24 --priority high
```

### Track Smart Money

```bash
# Find consistently profitable traders
insider-detector smart-money --platform polymarket --limit 20

# Watch a specific account
insider-detector watch 0x1234... --platform polymarket --reason "High win rate on political markets"
```

## Detection Methods

### 1. Win Rate Analysis
Accounts with win rates significantly above 50% on binary markets, especially on time-sensitive events, may have access to inside information.

### 2. Timing Analysis
Trades made shortly before market resolution or major price moves are flagged. Insiders often trade within 24-48 hours of announcements.

### 3. Position Concentration
Large concentrated positions on single markets suggest high conviction, potentially from private information.

### 4. Coordinated Trading
Multiple accounts trading in the same direction within a short time window may indicate coordinated insider activity.

### 5. Low-Probability Winners
Consistently winning bets on outcomes with <20% implied probability suggests either exceptional skill or inside information.

### 6. Account Clustering
Identifying groups of accounts that trade similarly across multiple markets helps detect sock puppet accounts.

## Configuration

Create a `.env` file for configuration:

```bash
# Kalshi API credentials (optional, required for some features)
KALSHI_API_KEY=your_api_key
KALSHI_API_SECRET=your_api_secret
KALSHI_USE_DEMO=false

# Database path
DATABASE_PATH=./data/insider_detector.db

# Debug mode
DEBUG=false
LOG_LEVEL=INFO
```

### Detection Thresholds

Edit `src/insider_detector/config.py` to adjust detection thresholds:

```python
@dataclass
class DetectionConfig:
    suspicious_hours_before_resolution: float = 48.0
    suspicious_win_rate: float = 0.75
    large_position_usd: float = 5000.0
    whale_position_usd: float = 50000.0
    min_trades_for_analysis: int = 5
```

## Known Insider Trading Cases

This tool is inspired by documented cases of insider trading on prediction markets:

1. **OpenAI Employees** - Traders with apparent inside knowledge of OpenAI announcements made large profits on Polymarket.

2. **Venezuela Invasion** - An account allegedly linked to Trump insiders made large bets on Venezuela-related markets before policy announcements.

3. **Supreme Court Decisions** - Unusual trading patterns before major court rulings suggested potential leaks.

## API Reference

### Python API

```python
import asyncio
from insider_detector import InsiderDetector
from insider_detector.models import Platform

async def main():
    async with InsiderDetector() as detector:
        # Scan markets
        result = await detector.scan_markets(
            platforms=[Platform.POLYMARKET],
            limit=50,
        )

        # Get suspicious accounts
        for account in result.suspicious_accounts:
            print(f"{account.id}: {account.suspicion_score:.2f}")

        # Get trading opportunities
        for alert in result.alerts:
            if alert.recommended_position:
                print(f"{alert.market_title}: {alert.recommended_position}")

asyncio.run(main())
```

### Real-time Monitoring

```python
async def handle_alert(alert):
    if alert.priority == "critical":
        print(f"CRITICAL: {alert.market_title}")
        print(f"Recommendation: {alert.recommended_position}")

async with InsiderDetector() as detector:
    await detector.monitor_live(
        platforms=[Platform.POLYMARKET],
        callback=handle_alert,
    )
```

## Database Schema

The tool uses SQLite for persistent storage:

- `markets` - Market metadata
- `trades` - Individual trades
- `accounts` - Account profiles and suspicion scores
- `alerts` - Generated alerts
- `watched_accounts` - Accounts being monitored

## Disclaimer

This tool is for **educational and research purposes only**.

- Past performance of detected accounts does not guarantee future results
- Not financial advice - always do your own research
- Some detected patterns may be coincidental
- Following "insider" accounts may not be profitable

Use at your own risk. The authors are not responsible for any financial losses.

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or PR.

Areas for improvement:
- WebSocket support for real-time trade streaming
- More sophisticated pattern detection (ML models)
- Integration with more prediction markets
- Browser extension for live monitoring
- Discord/Telegram bot for alerts
