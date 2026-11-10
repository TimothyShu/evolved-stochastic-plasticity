"""
infra/notify.py — ntfy push. No-op unless ESP_NTFY_TOPIC is set
(optionally ESP_NTFY_SERVER, default https://ntfy.sh). Never raises.
"""

from __future__ import annotations

import os
import urllib.request


def ntfy(message: str, title: str = "esp", priority: str = "default") -> bool:
    topic = os.environ.get("ESP_NTFY_TOPIC")
    if not topic:
        return False
    server = os.environ.get("ESP_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    req = urllib.request.Request(
        f"{server}/{topic}", data=message.encode(), method="POST",
        headers={"Title": title, "Priority": priority},
    )
    try:
        urllib.request.urlopen(req, timeout=10).close()
        return True
    except Exception as e:  # notification must never kill a run
        print(f"[ntfy] failed: {e}")
        return False
