"""公式サイトの直前情報ページから、展示タイムなどを取る。

展示航走はレースのおよそ20〜30分前に行われ、終わるとこのページに出る。
1レース1回のアクセスで済むので、見たいレースのときだけ取りにいく。
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd
import requests

URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"

# 艇番のセル → 選手名 → 体重 → 展示タイム → チルト の並び
ROW = re.compile(
    r'is-boatColor(\d) is-fs14" rowspan="4">\d</td>.*?'
    r'<td rowspan="2">([\d.]*)kg</td>\s*<td rowspan="4">([\d.]*|&nbsp;)</td>\s*<td rowspan="4">([-\d.]*|&nbsp;)</td>',
    re.S,
)


def _num(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def parse(html: str) -> pd.DataFrame:
    rows = [
        {"boat": int(b), "exhibition": _num(ex), "tilt": _num(tilt)}
        for b, _w, ex, tilt in ROW.findall(html)
    ]
    return pd.DataFrame(rows, columns=["boat", "exhibition", "tilt"])


def fetch(d: date, venue: str, race: int, session: requests.Session | None = None) -> pd.DataFrame:
    """展示がまだなら exhibition が空の表を返す。"""
    s = session or requests.Session()
    resp = s.get(
        URL, params={"rno": race, "jcd": venue, "hd": f"{d:%Y%m%d}"},
        headers={"User-Agent": "boatrace-yosou personal research"}, timeout=20,
    )
    resp.raise_for_status()
    return parse(resp.text)
