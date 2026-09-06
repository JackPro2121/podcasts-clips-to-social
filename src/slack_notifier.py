import os
import json
import logging
from typing import List, Dict, Any, Optional
import requests
from src.config import SLACK_WEBHOOK_URL, SLACK_BOT_TOKEN, SLACK_CHANNEL

logger = logging.getLogger(__name__)

class SlackNotifier:
    """
    Broadcasts operational reports and alert cards to Slack channels
    using Slack's Incoming Webhook or Bot Token chat.postMessage API.
    """
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        bot_token: Optional[str] = None,
        channel: Optional[str] = None
    ):
        self.webhook_url = webhook_url if webhook_url is not None else SLACK_WEBHOOK_URL
        self.bot_token = bot_token if bot_token is not None else SLACK_BOT_TOKEN
        self.channel = channel if channel is not None else (SLACK_CHANNEL or "podcast-clip")

    def is_enabled(self) -> bool:
        return bool((self.webhook_url and self.webhook_url.startswith("http")) or (self.bot_token and self.bot_token.startswith("xoxb-")))

    def send_run_report(
        self,
        podcast_title: str,
        podcast_url: str,
        niche: str,
        clips: List[Dict[str, Any]],
        release_url: Optional[str] = None
    ) -> bool:
        """
        Builds and sends a rich Slack Block Kit card summarizing the completed run.
        """
        if not self.is_enabled():
            print("[*] SlackNotifier: SLACK_WEBHOOK_URL not configured. Skipping Slack alert.")
            return False

        blocks: List[Dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🚀 Autonomous Podcast Clipper — Run Complete",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*🎙️ Podcast Episode:* <{podcast_url}|{podcast_title}>\n"
                        f"*🏷️ Niche:* `{niche}`  |  *🎬 Clips Generated:* `{len(clips)}`"
                    )
                }
            },
            {"type": "divider"}
        ]

        # Add clips breakdown
        for idx, clip in enumerate(clips, 1):
            title = clip.get("title", f"Clip #{idx}")
            score = clip.get("virality_score", 90)
            duration = clip.get("duration", 45)
            mp4_url = clip.get("download_url", "")
            buffer_status = clip.get("buffer_status", "Queued")
            due_at = clip.get("due_at", "")

            schedule_info = f"`{due_at}`" if due_at else f"`{buffer_status}`"
            link_md = f"<{mp4_url}|Download MP4>" if mp4_url else "Uploaded to Release"

            clip_text = (
                f"*#{idx} {title}*\n"
                f"• *Virality Score:* `{score}/100`  |  *Duration:* `{duration:.1f}s`\n"
                f"• *Buffer Status:* {schedule_info}  |  *Asset:* {link_md}"
            )

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": clip_text
                }
            })

        blocks.append({"type": "divider"})

        # Action Buttons / Links
        elements = []
        if release_url:
            elements.append({
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": "📦 GitHub Release Assets",
                    "emoji": True
                },
                "url": release_url,
                "action_id": "view_release"
            })
        elements.append({
            "type": "button",
            "text": {
                "type": "plain_text",
                "text": "🌐 View Source Episode",
                "emoji": True
            },
            "url": podcast_url,
            "action_id": "view_youtube"
        })

        if elements:
            blocks.append({
                "type": "actions",
                "elements": elements
            })

        # Context Footer
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "🤖 Autonomous AI Pipeline ($0 Budget) • Auto-cleanup active (>5 days)"
                }
            ]
        })

        return self._dispatch_blocks(blocks)

    def _dispatch_blocks(self, blocks: List[Dict[str, Any]]) -> bool:
        """Dispatches blocks via Incoming Webhook URL or Bot Token API."""
        try:
            if self.webhook_url and self.webhook_url.startswith("http"):
                res = requests.post(
                    self.webhook_url,
                    json={"blocks": blocks},
                    headers={"Content-Type": "application/json"},
                    timeout=15
                )
                if res.status_code == 200:
                    print(f"[+] Slack notification delivered successfully to #{self.channel} via Webhook!")
                    return True
                else:
                    print(f"[-] Slack webhook error: {res.status_code} - {res.text}")
                    return False

            elif self.bot_token and self.bot_token.startswith("xoxb-"):
                res = requests.post(
                    "https://slack.com/api/chat.postMessage",
                    json={
                        "channel": self.channel,
                        "blocks": blocks,
                        "text": "Autonomous Podcast Clipper Update"
                    },
                    headers={
                        "Authorization": f"Bearer {self.bot_token}",
                        "Content-Type": "application/json"
                    },
                    timeout=15
                )
                data = res.json()
                if data.get("ok"):
                    print(f"[+] Slack notification delivered successfully to #{self.channel} via Bot Token!")
                    return True
                else:
                    print(f"[-] Slack API error: {data.get('error')}")
                    return False
        except Exception as e:
            print(f"[-] Exception sending Slack notification: {e}")
            return False
        return False

    def send_error_alert(
        self,
        error_message: str,
        podcast_url: Optional[str] = None
    ) -> bool:
        """Sends high-priority red alert block if pipeline encounters an unhandled failure."""
        if not self.is_enabled():
            return False

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "⚠️ Autonomous Podcast Clipper Alert",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Pipeline Failure Detected!*\n"
                        f"*Source URL:* {podcast_url or 'N/A'}\n"
                        f"*Error:* ```{error_message}```"
                    )
                }
            }
        ]

        return self._dispatch_blocks(blocks)
