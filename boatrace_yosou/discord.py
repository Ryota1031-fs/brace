"""予想を Discord のチャンネルに投稿する（ウェブフック）。

ウェブフックのURLを知っている人は誰でもそのチャンネルに投稿できるので、URLはこのPCの data/discord.json にだけ保存する
（data/ は GitHub に上げず、アプリの更新でも消えない）。
投稿するのはアプリが計算した予想だけ。出走表などの公式のデータそのものは載せず、公式のレースページへのリンクを付ける。
"""
from __future__ import annotations

import json
import re
from datetime import date

import requests

from .download import DATA_DIR

CONFIG = DATA_DIR.parent / "discord.json"
URL_PATTERN = re.compile(r"^https://(?:canary\.|ptb\.)?discord(?:app)?\.com/api/webhooks/\d+/[\w-]+$")
BOAT_MARKS = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣", 6: "6️⃣"}
RACE_PAGE = "https://www.boatrace.jp/owpc/pc/race/racelist?rno={race}&jcd={venue}&hd={d:%Y%m%d}"


def load_url() -> str:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8")).get("webhook_url", "")
    except (OSError, ValueError):
        return ""


def save_url(url: str) -> None:
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"webhook_url": url.strip()}), encoding="utf-8")


def valid_url(url: str) -> bool:
    return bool(URL_PATTERN.match(url.strip()))


def _combo(c: str) -> str:
    return "".join(BOAT_MARKS[int(b)] for b in c.split("-"))


def format_message(d: date, venue: str, venue_name: str, race: int, model: str, has_ex: bool,
                   win_probs: dict[int, float], picks: dict[str, list[dict]], values: list[dict]) -> str:
    """picks は {券種: [{combo, prob, odds}]}、values は妙味のある買い目 [{ticket, combo, prob, odds, ev}]。"""
    lines = [
        f"🚤 **{venue_name} {race}R**（{d.month}/{d.day}）の予想",
        f"{model}・{'展示タイム反映済み' if has_ex else '展示タイム前'}",
        "",
        "**1着確率**",
        "  ".join(f"{BOAT_MARKS[b]} {p:.0%}" for b, p in sorted(win_probs.items())),
        "",
        "**おすすめの買い目**",
    ]
    for t, xs in picks.items():
        lines.append(f"{t}：" + "　".join(f"{_combo(x['combo'])} {x['prob']:.1%}" for x in xs))
    if values:
        lines += ["", "**妙味のある買い目**（確率 × オッズ = 期待値）"]
        for x in values:
            lines.append(f"{x['ticket']} {_combo(x['combo'])} {x['prob']:.1%} × {x['odds']:.1f}倍 = {x['ev']:.2f}")
    lines += ["", f"出走表: <{RACE_PAGE.format(race=race, venue=venue, d=d)}>",
              "※確率は過去の傾向から計算した目安です。舟券は無理のない範囲で。"]
    return "\n".join(lines)[:2000]  # Discord の1投稿は2000文字まで


def send(url: str, content: str) -> None:
    resp = requests.post(url, json={"content": content, "username": "競艇予想"}, timeout=15)
    resp.raise_for_status()
