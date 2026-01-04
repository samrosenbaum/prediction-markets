"""
FastAPI server for the insider trading detector dashboard.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import get_config
from ..detection.detector import InsiderDetector
from ..detection.alerts import AlertManager
from ..models import Platform
from ..storage.database import Database

logger = logging.getLogger(__name__)

# Global state
detector: Optional[InsiderDetector] = None
db: Optional[Database] = None
alert_manager: Optional[AlertManager] = None
background_scan_running = False


# Pydantic models for API responses
class StatsResponse(BaseModel):
    markets: int
    trades: int
    accounts: int
    suspicious_accounts: int
    alerts: int
    last_scan: Optional[str] = None


class AccountResponse(BaseModel):
    id: str
    platform: str
    address: str
    total_trades: int
    total_markets: int
    total_volume: float
    total_pnl: float
    win_rate: float
    suspicion_score: float
    suspicion_reasons: list[str]
    first_seen: Optional[str]
    last_seen: Optional[str]


class AlertResponse(BaseModel):
    id: str
    timestamp: str
    platform: str
    market_id: str
    market_title: str
    alert_type: str
    priority: str
    suspicious_accounts: list[str]
    total_suspicious_volume: float
    current_price: float
    recommended_position: str
    confidence: float
    reasoning: str


class MarketResponse(BaseModel):
    id: str
    platform: str
    title: str
    status: str
    volume: float
    current_yes_price: float
    url: str


class OpportunityResponse(BaseModel):
    market_id: str
    market_title: str
    platform: str
    recommended_position: str
    confidence: float
    current_price: float
    reasoning: str
    priority: str
    suspicious_accounts: int
    volume: float


class ScanRequest(BaseModel):
    platforms: list[str] = ["polymarket"]
    limit: int = 50


class ScanResponse(BaseModel):
    status: str
    message: str
    suspicious_accounts: int = 0
    alerts: int = 0


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, db, alert_manager

    # Startup
    logger.info("Starting dashboard server...")
    detector = InsiderDetector()
    await detector.connect()

    db = Database()
    await db.connect()

    alert_manager = AlertManager()

    yield

    # Shutdown
    logger.info("Shutting down dashboard server...")
    if detector:
        await detector.close()
    if db:
        await db.close()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="Insider Trading Detector",
        description="Dashboard for detecting insider trading on prediction markets",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routes
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main dashboard page."""
        html_file = Path(__file__).parent / "static" / "index.html"
        if html_file.exists():
            return HTMLResponse(content=html_file.read_text())
        return HTMLResponse(content="<h1>Dashboard</h1><p>Static files not found</p>")

    @app.get("/api/stats", response_model=StatsResponse)
    async def get_stats():
        """Get dashboard statistics."""
        stats = await db.get_stats()
        return StatsResponse(**stats)

    @app.get("/api/accounts", response_model=list[AccountResponse])
    async def get_accounts(
        min_score: float = Query(0.0, ge=0, le=1),
        limit: int = Query(100, ge=1, le=500),
        platform: Optional[str] = None,
    ):
        """Get suspicious accounts."""
        accounts = await db.get_suspicious_accounts(min_score=min_score, limit=limit)

        if platform:
            accounts = [a for a in accounts if a.platform.value == platform]

        return [
            AccountResponse(
                id=a.id,
                platform=a.platform.value,
                address=a.address,
                total_trades=a.total_trades,
                total_markets=a.total_markets,
                total_volume=float(a.total_volume),
                total_pnl=float(a.total_pnl),
                win_rate=a.win_rate,
                suspicion_score=a.suspicion_score,
                suspicion_reasons=a.suspicion_reasons,
                first_seen=a.first_seen.isoformat() if a.first_seen else None,
                last_seen=a.last_seen.isoformat() if a.last_seen else None,
            )
            for a in accounts
        ]

    @app.get("/api/accounts/{account_id}")
    async def get_account(account_id: str, platform: str = "polymarket"):
        """Get detailed account information."""
        plat = Platform.POLYMARKET if platform == "polymarket" else Platform.KALSHI

        # Get from database first
        accounts = await db.get_suspicious_accounts(min_score=0, limit=1000)
        account = next((a for a in accounts if a.id == account_id), None)

        if not account:
            # Fetch live
            try:
                result = await detector.scan_account(plat, account_id)
                if result.suspicious_accounts:
                    account = result.suspicious_accounts[0]
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Account not found: {e}")

        if not account:
            raise HTTPException(status_code=404, detail="Account not found")

        # Get trades
        trades = await db.get_trades(account_id=account_id, limit=100)

        return {
            "account": AccountResponse(
                id=account.id,
                platform=account.platform.value,
                address=account.address,
                total_trades=account.total_trades,
                total_markets=account.total_markets,
                total_volume=float(account.total_volume),
                total_pnl=float(account.total_pnl),
                win_rate=account.win_rate,
                suspicion_score=account.suspicion_score,
                suspicion_reasons=account.suspicion_reasons,
                first_seen=account.first_seen.isoformat() if account.first_seen else None,
                last_seen=account.last_seen.isoformat() if account.last_seen else None,
            ),
            "trades": [
                {
                    "id": t.id,
                    "market_id": t.market_id,
                    "direction": t.direction.value,
                    "is_yes": t.is_yes,
                    "price": float(t.price),
                    "size": float(t.size),
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in trades
            ],
        }

    @app.get("/api/alerts", response_model=list[AlertResponse])
    async def get_alerts(
        hours: int = Query(24, ge=1, le=168),
        priority: Optional[str] = None,
        limit: int = Query(100, ge=1, le=500),
    ):
        """Get recent alerts."""
        alerts = await db.get_recent_alerts(hours=hours, priority=priority, limit=limit)

        return [
            AlertResponse(
                id=a.id,
                timestamp=a.timestamp.isoformat(),
                platform=a.platform.value,
                market_id=a.market_id,
                market_title=a.market_title,
                alert_type=a.alert_type,
                priority=a.priority,
                suspicious_accounts=a.suspicious_accounts,
                total_suspicious_volume=float(a.total_suspicious_volume),
                current_price=float(a.current_price),
                recommended_position=a.recommended_position,
                confidence=a.confidence,
                reasoning=a.reasoning,
            )
            for a in alerts
        ]

    @app.get("/api/opportunities", response_model=list[OpportunityResponse])
    async def get_opportunities(
        min_confidence: float = Query(0.5, ge=0, le=1),
        platform: Optional[str] = None,
    ):
        """Get current trading opportunities."""
        alerts = await db.get_recent_alerts(hours=48, priority=None, limit=100)

        opportunities = []
        for alert in alerts:
            if alert.confidence >= min_confidence and alert.recommended_position:
                if platform and alert.platform.value != platform:
                    continue
                opportunities.append(
                    OpportunityResponse(
                        market_id=alert.market_id,
                        market_title=alert.market_title,
                        platform=alert.platform.value,
                        recommended_position=alert.recommended_position,
                        confidence=alert.confidence,
                        current_price=float(alert.current_price),
                        reasoning=alert.reasoning,
                        priority=alert.priority,
                        suspicious_accounts=len(alert.suspicious_accounts),
                        volume=float(alert.total_suspicious_volume),
                    )
                )

        # Sort by confidence
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        return opportunities

    @app.post("/api/scan", response_model=ScanResponse)
    async def start_scan(request: ScanRequest, background_tasks: BackgroundTasks):
        """Start a market scan."""
        global background_scan_running

        if background_scan_running:
            return ScanResponse(
                status="running",
                message="A scan is already in progress",
            )

        async def run_scan():
            global background_scan_running
            background_scan_running = True

            try:
                platforms = [
                    Platform.POLYMARKET if p == "polymarket" else Platform.KALSHI
                    for p in request.platforms
                ]

                result = await detector.scan_markets(
                    platforms=platforms,
                    limit=request.limit,
                )

                # Save results
                for acc in result.suspicious_accounts:
                    await db.save_account(acc)
                for alert in result.alerts:
                    await db.save_alert(alert)

                logger.info(
                    f"Scan complete: {len(result.suspicious_accounts)} suspicious accounts, "
                    f"{len(result.alerts)} alerts"
                )

            except Exception as e:
                logger.error(f"Scan failed: {e}")
            finally:
                background_scan_running = False

        background_tasks.add_task(run_scan)

        return ScanResponse(
            status="started",
            message=f"Scanning {request.limit} markets on {', '.join(request.platforms)}",
        )

    @app.get("/api/scan/status")
    async def get_scan_status():
        """Get current scan status."""
        return {"running": background_scan_running}

    @app.post("/api/watch/{account_id}")
    async def watch_account(
        account_id: str,
        platform: str = "polymarket",
        reason: str = "",
    ):
        """Add an account to the watch list."""
        plat = Platform.POLYMARKET if platform == "polymarket" else Platform.KALSHI
        await db.add_watched_account(plat, account_id, reason)
        return {"status": "success", "message": f"Now watching {account_id}"}

    @app.get("/api/watched")
    async def get_watched_accounts():
        """Get all watched accounts."""
        accounts = await db.get_watched_accounts()
        return [
            {"account_id": a[0], "platform": a[1].value, "reason": a[2]}
            for a in accounts
        ]

    @app.get("/api/markets")
    async def get_markets(
        platform: str = "polymarket",
        status: str = "active",
        limit: int = 50,
    ):
        """Get markets from the platform."""
        plat = Platform.POLYMARKET if platform == "polymarket" else Platform.KALSHI
        client = detector.get_client(plat)

        markets = await client.get_markets(status=status, limit=limit)

        return [
            MarketResponse(
                id=m.id,
                platform=m.platform.value,
                title=m.title,
                status=m.status.value,
                volume=float(m.volume),
                current_yes_price=float(m.current_yes_price),
                url=m.url,
            )
            for m in markets
        ]

    @app.get("/api/market/{market_id}/trades")
    async def get_market_trades(market_id: str, platform: str = "polymarket", limit: int = 100):
        """Get trades for a specific market."""
        trades = await db.get_trades(market_id=market_id, limit=limit)

        return [
            {
                "id": t.id,
                "account_id": t.account_id,
                "direction": t.direction.value,
                "is_yes": t.is_yes,
                "price": float(t.price),
                "size": float(t.size),
                "timestamp": t.timestamp.isoformat(),
            }
            for t in trades
        ]

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the dashboard server."""
    import uvicorn

    uvicorn.run(
        "insider_detector.dashboard.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )
