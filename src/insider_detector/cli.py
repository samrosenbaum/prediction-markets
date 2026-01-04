"""
Command-line interface for the insider trading detector.

Usage:
    insider-detector scan --platform polymarket --limit 50
    insider-detector monitor --platform polymarket,kalshi
    insider-detector account 0x1234... --platform polymarket
    insider-detector opportunities
    insider-detector alerts --priority high
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import Config, get_config, set_config
from .detection.alerts import AlertManager, ConsoleNotifier
from .detection.detector import InsiderDetector
from .models import Platform
from .storage.database import Database

app = typer.Typer(
    name="insider-detector",
    help="Detect potential insider trading on prediction markets (Polymarket, Kalshi)",
    add_completion=False,
)

console = Console()


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Quiet noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_platforms(platforms: str) -> list[Platform]:
    """Parse comma-separated platform names."""
    result = []
    for p in platforms.lower().split(","):
        p = p.strip()
        if p == "polymarket":
            result.append(Platform.POLYMARKET)
        elif p == "kalshi":
            result.append(Platform.KALSHI)
    return result or [Platform.POLYMARKET]


@app.command()
def scan(
    platforms: str = typer.Option(
        "polymarket",
        "--platform", "-p",
        help="Platforms to scan (comma-separated: polymarket,kalshi)",
    ),
    limit: int = typer.Option(
        50,
        "--limit", "-l",
        help="Maximum number of markets to scan",
    ),
    min_suspicion: float = typer.Option(
        0.5,
        "--min-suspicion", "-s",
        help="Minimum suspicion score to report (0-1)",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save results to database",
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """
    Scan markets for suspicious trading activity.

    This will fetch recent trades, analyze accounts, and detect
    potential insider trading patterns.
    """
    setup_logging(debug)

    platform_list = parse_platforms(platforms)

    async def run_scan():
        console.print(Panel(
            f"[bold]Scanning {limit} markets on {', '.join(p.value for p in platform_list)}[/bold]",
            title="Insider Trading Detector",
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Connecting to APIs...", total=None)

            async with InsiderDetector() as detector:
                progress.update(task, description="Scanning markets...")

                result = await detector.scan_markets(
                    platforms=platform_list,
                    limit=limit,
                )

                progress.update(task, description="Analyzing results...")

        # Display results
        console.print()

        if result.suspicious_accounts:
            table = Table(title="Suspicious Accounts", show_header=True)
            table.add_column("Account", style="cyan")
            table.add_column("Platform")
            table.add_column("Score", justify="right")
            table.add_column("Win Rate", justify="right")
            table.add_column("P&L", justify="right")
            table.add_column("Reasons")

            for acc in result.suspicious_accounts[:20]:
                if acc.suspicion_score >= min_suspicion:
                    table.add_row(
                        acc.id[:16] + "..." if len(acc.id) > 16 else acc.id,
                        acc.platform.value,
                        f"{acc.suspicion_score:.2f}",
                        f"{acc.win_rate:.1%}",
                        f"${float(acc.total_pnl):,.0f}",
                        ", ".join(acc.suspicion_reasons[:2]),
                    )

            console.print(table)
        else:
            console.print("[yellow]No suspicious accounts found[/yellow]")

        if result.alerts:
            console.print()
            table = Table(title="Alerts", show_header=True)
            table.add_column("Priority", style="bold")
            table.add_column("Type")
            table.add_column("Market")
            table.add_column("Accounts")
            table.add_column("Recommendation")

            for alert in result.alerts[:10]:
                priority_style = {
                    "critical": "red bold",
                    "high": "red",
                    "medium": "yellow",
                    "low": "dim",
                }.get(alert.priority, "")

                table.add_row(
                    f"[{priority_style}]{alert.priority.upper()}[/]",
                    alert.alert_type,
                    alert.market_title[:30] + "..." if len(alert.market_title) > 30 else alert.market_title,
                    str(len(alert.suspicious_accounts)),
                    f"{alert.recommended_position} ({alert.confidence:.0%})" if alert.recommended_position else "-",
                )

            console.print(table)

        # Summary
        console.print()
        console.print(Panel(
            f"Found [bold]{len(result.suspicious_accounts)}[/bold] suspicious accounts, "
            f"[bold]{len(result.alerts)}[/bold] alerts, "
            f"[bold]{len(result.anomaly_signals)}[/bold] anomalies",
            title="Summary",
        ))

        # Save to database
        if save:
            async with Database() as db:
                for acc in result.suspicious_accounts:
                    await db.save_account(acc)
                for alert in result.alerts:
                    await db.save_alert(alert)
                console.print("[dim]Results saved to database[/dim]")

    asyncio.run(run_scan())


@app.command()
def monitor(
    platforms: str = typer.Option(
        "polymarket",
        "--platform", "-p",
        help="Platforms to monitor",
    ),
    markets: Optional[str] = typer.Option(
        None,
        "--markets", "-m",
        help="Specific market IDs to watch (comma-separated)",
    ),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    """
    Monitor live trades in real-time for suspicious activity.

    Press Ctrl+C to stop monitoring.
    """
    setup_logging(debug)

    platform_list = parse_platforms(platforms)
    market_ids = markets.split(",") if markets else None

    alert_manager = AlertManager()
    ConsoleNotifier(alert_manager)

    async def handle_alert(alert):
        await alert_manager.process_alert(alert)

    async def run_monitor():
        console.print(Panel(
            f"[bold]Monitoring live trades on {', '.join(p.value for p in platform_list)}[/bold]\n"
            f"Press Ctrl+C to stop",
            title="Live Monitor",
        ))

        try:
            async with InsiderDetector() as detector:
                await detector.monitor_live(
                    platforms=platform_list,
                    market_ids=market_ids,
                    callback=handle_alert,
                )
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped[/yellow]")

    try:
        asyncio.run(run_monitor())
    except KeyboardInterrupt:
        pass


@app.command()
def account(
    account_id: str = typer.Argument(..., help="Account ID or wallet address"),
    platform: str = typer.Option(
        "polymarket",
        "--platform", "-p",
        help="Platform (polymarket or kalshi)",
    ),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    """
    Analyze a specific account for suspicious activity.
    """
    setup_logging(debug)

    plat = Platform.POLYMARKET if platform.lower() == "polymarket" else Platform.KALSHI

    async def run_analysis():
        console.print(f"[bold]Analyzing account: {account_id}[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching account data...", total=None)

            async with InsiderDetector() as detector:
                result = await detector.scan_account(plat, account_id)

        if not result.suspicious_accounts and not result.patterns:
            console.print("[yellow]No significant activity found[/yellow]")
            return

        # Display profile
        for acc in result.suspicious_accounts:
            console.print()
            console.print(Panel(
                f"[bold]Account:[/bold] {acc.id}\n"
                f"[bold]Platform:[/bold] {acc.platform.value}\n"
                f"[bold]Total Trades:[/bold] {acc.total_trades}\n"
                f"[bold]Win Rate:[/bold] {acc.win_rate:.1%}\n"
                f"[bold]Total P&L:[/bold] ${float(acc.total_pnl):,.2f}\n"
                f"[bold]Suspicion Score:[/bold] {acc.suspicion_score:.2f}\n"
                f"[bold]Reasons:[/bold] {', '.join(acc.suspicion_reasons)}",
                title="Account Profile",
            ))

        # Display patterns
        if result.patterns:
            console.print()
            table = Table(title="Detected Patterns", show_header=True)
            table.add_column("Pattern")
            table.add_column("Confidence")
            table.add_column("Description")

            for pattern in result.patterns:
                table.add_row(
                    pattern.pattern_type,
                    f"{pattern.confidence:.1%}",
                    pattern.description,
                )

            console.print(table)

    asyncio.run(run_analysis())


@app.command()
def opportunities(
    platforms: str = typer.Option(
        "polymarket",
        "--platform", "-p",
        help="Platforms to check",
    ),
    min_confidence: float = typer.Option(
        0.5,
        "--min-confidence", "-c",
        help="Minimum confidence threshold",
    ),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    """
    Show current trading opportunities based on detected insider activity.

    These are markets where suspicious accounts are taking positions,
    suggesting potential informed trading.
    """
    setup_logging(debug)

    platform_list = parse_platforms(platforms)

    async def run():
        console.print("[bold]Finding trading opportunities...[/bold]")

        async with Database() as db:
            alerts = await db.get_recent_alerts(hours=48, priority="high")

        if not alerts:
            # Run a quick scan
            async with InsiderDetector() as detector:
                result = await detector.scan_markets(platforms=platform_list, limit=30)
                alerts = result.alerts

        if not alerts:
            console.print("[yellow]No opportunities found. Run 'scan' first to analyze markets.[/yellow]")
            return

        # Filter and display
        opportunities = []
        for alert in alerts:
            if alert.confidence >= min_confidence and alert.recommended_position:
                opportunities.append(alert)

        if not opportunities:
            console.print(f"[yellow]No opportunities with confidence >= {min_confidence:.0%}[/yellow]")
            return

        table = Table(title="Trading Opportunities", show_header=True)
        table.add_column("Market", style="cyan", max_width=40)
        table.add_column("Platform")
        table.add_column("Position", style="bold")
        table.add_column("Price", justify="right")
        table.add_column("Confidence", justify="right")
        table.add_column("Signal")

        for opp in sorted(opportunities, key=lambda x: x.confidence, reverse=True)[:15]:
            pos_style = "green" if opp.recommended_position == "YES" else "red"
            table.add_row(
                opp.market_title[:40] if opp.market_title else opp.market_id[:40],
                opp.platform.value,
                f"[{pos_style}]{opp.recommended_position}[/]",
                f"{float(opp.current_price):.1%}",
                f"{opp.confidence:.0%}",
                opp.alert_type,
            )

        console.print(table)

        console.print()
        console.print("[dim]These opportunities are based on detected suspicious activity. "
                     "Always do your own research before trading.[/dim]")

    asyncio.run(run())


@app.command()
def alerts(
    hours: int = typer.Option(24, "--hours", "-h", help="Look back period in hours"),
    priority: Optional[str] = typer.Option(None, "--priority", "-p", help="Filter by priority"),
    export: Optional[Path] = typer.Option(None, "--export", "-e", help="Export to JSON file"),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    """
    View recent alerts.
    """
    setup_logging(debug)

    async def run():
        async with Database() as db:
            recent_alerts = await db.get_recent_alerts(hours=hours, priority=priority)

        if not recent_alerts:
            console.print(f"[yellow]No alerts in the last {hours} hours[/yellow]")
            return

        table = Table(title=f"Alerts (Last {hours} Hours)", show_header=True)
        table.add_column("Time")
        table.add_column("Priority", style="bold")
        table.add_column("Type")
        table.add_column("Market", max_width=30)
        table.add_column("Accounts")
        table.add_column("Reasoning", max_width=40)

        for alert in recent_alerts[:30]:
            priority_style = {
                "critical": "red bold",
                "high": "red",
                "medium": "yellow",
                "low": "dim",
            }.get(alert.priority, "")

            table.add_row(
                alert.timestamp.strftime("%m-%d %H:%M"),
                f"[{priority_style}]{alert.priority.upper()}[/]",
                alert.alert_type,
                alert.market_title[:30] if alert.market_title else alert.market_id[:30],
                str(len(alert.suspicious_accounts)),
                alert.reasoning[:40] + "..." if len(alert.reasoning) > 40 else alert.reasoning,
            )

        console.print(table)

        if export:
            manager = AlertManager()
            for alert in recent_alerts:
                await manager.process_alert(alert)
            manager.export_alerts(export)
            console.print(f"[green]Exported to {export}[/green]")

    asyncio.run(run())


@app.command()
def watch(
    account_id: str = typer.Argument(..., help="Account ID to watch"),
    platform: str = typer.Option("polymarket", "--platform", "-p"),
    reason: str = typer.Option("", "--reason", "-r", help="Reason for watching"),
):
    """
    Add an account to the watch list.

    Watched accounts will trigger alerts when they trade.
    """
    plat = Platform.POLYMARKET if platform.lower() == "polymarket" else Platform.KALSHI

    async def run():
        async with Database() as db:
            await db.add_watched_account(plat, account_id, reason)

        console.print(f"[green]Added {account_id} to watch list[/green]")

    asyncio.run(run())


@app.command()
def stats():
    """
    Show database statistics.
    """
    async def run():
        async with Database() as db:
            stats = await db.get_stats()

        console.print(Panel(
            f"[bold]Markets tracked:[/bold] {stats['markets']}\n"
            f"[bold]Trades stored:[/bold] {stats['trades']}\n"
            f"[bold]Accounts analyzed:[/bold] {stats['accounts']}\n"
            f"[bold]Suspicious accounts:[/bold] {stats['suspicious_accounts']}\n"
            f"[bold]Alerts generated:[/bold] {stats['alerts']}",
            title="Database Statistics",
        ))

    asyncio.run(run())


@app.command()
def smart_money(
    platform: str = typer.Option("polymarket", "--platform", "-p"),
    limit: int = typer.Option(20, "--limit", "-l"),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    """
    Find consistently profitable traders (smart money) to follow.
    """
    setup_logging(debug)

    plat = Platform.POLYMARKET if platform.lower() == "polymarket" else Platform.KALSHI

    async def run():
        console.print("[bold]Finding smart money accounts...[/bold]")

        async with InsiderDetector() as detector:
            traders = await detector.find_smart_money(plat, limit=limit)

        if not traders:
            console.print("[yellow]No smart money accounts found[/yellow]")
            return

        table = Table(title="Smart Money Accounts", show_header=True)
        table.add_column("Account", style="cyan")
        table.add_column("Win Rate", justify="right")
        table.add_column("Total P&L", justify="right", style="green")
        table.add_column("Volume", justify="right")
        table.add_column("Trades", justify="right")

        for trader in traders[:20]:
            table.add_row(
                trader.id[:20] + "..." if len(trader.id) > 20 else trader.id,
                f"{trader.win_rate:.1%}",
                f"${float(trader.total_pnl):,.0f}",
                f"${float(trader.total_volume):,.0f}",
                str(trader.total_trades),
            )

        console.print(table)

        console.print()
        console.print("[dim]Consider watching these accounts for trading signals[/dim]")

    asyncio.run(run())


@app.command()
def dashboard(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development"),
    debug: bool = typer.Option(False, "--debug", "-d"),
):
    """
    Start the web dashboard.

    Opens a browser-based interface for analyzing suspicious accounts,
    viewing alerts, and finding trading opportunities.
    """
    setup_logging(debug)

    console.print(Panel(
        f"[bold]Starting dashboard server[/bold]\n\n"
        f"Open [cyan]http://localhost:{port}[/cyan] in your browser\n\n"
        f"Press Ctrl+C to stop",
        title="Insider Trading Detector Dashboard",
    ))

    try:
        from .dashboard import run_server
        run_server(host=host, port=port, reload=reload)
    except ImportError as e:
        console.print(f"[red]Failed to start dashboard: {e}[/red]")
        console.print("[yellow]Make sure to install dashboard dependencies: pip install fastapi uvicorn[/yellow]")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
