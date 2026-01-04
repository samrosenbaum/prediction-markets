"""
Notification services for sending alerts via Telegram and Discord.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

import httpx

from ..models import Alert, Platform

logger = logging.getLogger(__name__)


class NotificationService(ABC):
    """Base class for notification services."""

    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert. Returns True if successful."""
        pass

    @abstractmethod
    async def send_message(self, message: str) -> bool:
        """Send a plain text message."""
        pass


@dataclass
class TelegramConfig:
    """Configuration for Telegram bot."""
    bot_token: str
    chat_id: str  # Can be a user ID, group ID, or channel username
    parse_mode: str = "HTML"
    disable_notification: bool = False  # Silent notifications


class TelegramNotifier(NotificationService):
    """
    Send alerts via Telegram bot.

    To set up:
    1. Create a bot with @BotFather on Telegram
    2. Get the bot token
    3. Start a chat with your bot and get your chat_id
       (send a message, then check https://api.telegram.org/bot<TOKEN>/getUpdates)
    """

    def __init__(self, config: TelegramConfig):
        self.config = config
        self.api_base = f"https://api.telegram.org/bot{config.bot_token}"

    async def send_message(self, message: str) -> bool:
        """Send a plain text message."""
        url = f"{self.api_base}/sendMessage"
        payload = {
            "chat_id": self.config.chat_id,
            "text": message,
            "parse_mode": self.config.parse_mode,
            "disable_notification": self.config.disable_notification,
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    return True
                else:
                    logger.error(f"Telegram API error: {resp.status_code} - {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def send_alert(self, alert: Alert) -> bool:
        """Send a formatted alert."""
        message = self._format_alert(alert)
        return await self.send_message(message)

    def _format_alert(self, alert: Alert) -> str:
        """Format an alert for Telegram (HTML)."""
        priority_emoji = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "📢",
            "low": "📌",
        }

        emoji = priority_emoji.get(alert.priority, "📌")

        lines = [
            f"{emoji} <b>[{alert.priority.upper()}] {alert.alert_type}</b>",
            f"",
            f"<b>Market:</b> {self._escape_html(alert.market_title or alert.market_id)}",
            f"<b>Platform:</b> {alert.platform.value}",
            f"<b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
        ]

        if alert.recommended_position:
            position_emoji = "🟢" if alert.recommended_position == "YES" else "🔴"
            lines.append(f"")
            lines.append(f"{position_emoji} <b>Signal: {alert.recommended_position}</b> @ {float(alert.current_price):.1%}")
            lines.append(f"<b>Confidence:</b> {alert.confidence:.0%}")

        if alert.reasoning:
            lines.append(f"")
            lines.append(f"<i>{self._escape_html(alert.reasoning[:200])}</i>")

        if alert.suspicious_accounts:
            lines.append(f"")
            lines.append(f"👥 {len(alert.suspicious_accounts)} suspicious accounts")

        if alert.total_suspicious_volume > 0:
            lines.append(f"💰 ${float(alert.total_suspicious_volume):,.0f} volume")

        return "\n".join(lines)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


@dataclass
class DiscordConfig:
    """Configuration for Discord webhook."""
    webhook_url: str
    username: str = "Insider Trading Detector"
    avatar_url: Optional[str] = None


class DiscordNotifier(NotificationService):
    """
    Send alerts via Discord webhook.

    To set up:
    1. In your Discord server, go to Server Settings > Integrations > Webhooks
    2. Create a new webhook and copy the URL
    """

    def __init__(self, config: DiscordConfig):
        self.config = config

    async def send_message(self, message: str) -> bool:
        """Send a plain text message."""
        payload = {
            "content": message,
            "username": self.config.username,
        }
        if self.config.avatar_url:
            payload["avatar_url"] = self.config.avatar_url

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=10,
                )
                if resp.status_code in (200, 204):
                    return True
                else:
                    logger.error(f"Discord webhook error: {resp.status_code} - {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Discord message: {e}")
            return False

    async def send_alert(self, alert: Alert) -> bool:
        """Send a formatted alert as Discord embed."""
        embed = self._create_embed(alert)
        payload = {
            "username": self.config.username,
            "embeds": [embed],
        }
        if self.config.avatar_url:
            payload["avatar_url"] = self.config.avatar_url

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=10,
                )
                if resp.status_code in (200, 204):
                    return True
                else:
                    logger.error(f"Discord webhook error: {resp.status_code} - {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    def _create_embed(self, alert: Alert) -> dict:
        """Create a Discord embed for an alert."""
        color_map = {
            "critical": 0xFF0000,  # Red
            "high": 0xFFA500,      # Orange
            "medium": 0xFFFF00,    # Yellow
            "low": 0x808080,       # Gray
        }

        fields = [
            {"name": "Platform", "value": alert.platform.value, "inline": True},
            {"name": "Alert Type", "value": alert.alert_type, "inline": True},
            {"name": "Priority", "value": alert.priority.upper(), "inline": True},
        ]

        if alert.recommended_position:
            fields.append({
                "name": "📊 Signal",
                "value": f"**{alert.recommended_position}** @ {float(alert.current_price):.1%}",
                "inline": True,
            })
            fields.append({
                "name": "Confidence",
                "value": f"{alert.confidence:.0%}",
                "inline": True,
            })

        if alert.suspicious_accounts:
            fields.append({
                "name": "👥 Suspicious Accounts",
                "value": str(len(alert.suspicious_accounts)),
                "inline": True,
            })

        if alert.total_suspicious_volume > 0:
            fields.append({
                "name": "💰 Volume",
                "value": f"${float(alert.total_suspicious_volume):,.0f}",
                "inline": True,
            })

        embed = {
            "title": f"🔔 {alert.market_title or alert.market_id}",
            "description": alert.reasoning[:500] if alert.reasoning else None,
            "color": color_map.get(alert.priority, 0x808080),
            "fields": fields,
            "timestamp": alert.timestamp.isoformat(),
            "footer": {"text": "Insider Trading Detector"},
        }

        return {k: v for k, v in embed.items() if v is not None}


class MultiNotifier(NotificationService):
    """Send alerts to multiple notification services."""

    def __init__(self, services: list[NotificationService]):
        self.services = services

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to all services."""
        if not self.services:
            return False

        results = await asyncio.gather(
            *[service.send_alert(alert) for service in self.services],
            return_exceptions=True,
        )

        # Return True if at least one succeeded
        return any(r is True for r in results)

    async def send_message(self, message: str) -> bool:
        """Send message to all services."""
        if not self.services:
            return False

        results = await asyncio.gather(
            *[service.send_message(message) for service in self.services],
            return_exceptions=True,
        )

        return any(r is True for r in results)


def create_notifier_from_env() -> Optional[MultiNotifier]:
    """
    Create a notifier from environment variables.

    Environment variables:
        TELEGRAM_BOT_TOKEN: Telegram bot token
        TELEGRAM_CHAT_ID: Telegram chat/user ID
        DISCORD_WEBHOOK_URL: Discord webhook URL
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    services = []

    # Telegram
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID")
    if telegram_token and telegram_chat:
        config = TelegramConfig(bot_token=telegram_token, chat_id=telegram_chat)
        services.append(TelegramNotifier(config))
        logger.info("Telegram notifications enabled")

    # Discord
    discord_url = os.getenv("DISCORD_WEBHOOK_URL")
    if discord_url:
        config = DiscordConfig(webhook_url=discord_url)
        services.append(DiscordNotifier(config))
        logger.info("Discord notifications enabled")

    if services:
        return MultiNotifier(services)

    return None
