"""公式サイトのオッズページから、2連単・2連複・3連単・3連複のオッズを取る。

オッズ表は「1着（列）× 2着・3着（行）」の格子になっている。列は1号艇〜6号艇の順で、
各列は [2着（3連系は同じ2着の行のまとまりの先頭だけ）] [3着] [オッズ] のセルの並び。出せない組（3連複の重複など）は空セル。
"""
from __future__ import annotations

import re
import time
from datetime import date

import requests

URL = "https://www.boatrace.jp/owpc/pc/race/{page}"
PAGES = {"odds3t": ["3連単"], "odds3f": ["3連複"], "odds2tf": ["2連単", "2連複"]}

TABLE = re.compile(r"<tbody[^>]*>(.*?)</tbody>", re.S)
ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
CELL = re.compile(r"<td([^>]*)>(.*?)</td>", re.S)


def _num(s: str):
    try:
        return float(re.sub(r"<[^>]+>", "", s).strip())
    except ValueError:
        return None


def _grid(body: str, three: bool) -> list[tuple[int, int, int | None, float | None]]:
    """(1着, 2着, 3着, オッズ) の並びを返す。2連系は3着が None。"""
    out = []
    second = [None] * 6
    for ri, row in enumerate(ROW.findall(body)):
        cells = CELL.findall(row)
        # 2着のセルがある行は1列3セル、ない行は2セル（2着は上の行から引き継ぐ）
        has_second = three and len(cells) == 18
        i = 0
        for col in range(6):
            if three:
                if has_second:
                    second[col] = _num(cells[i][1])
                    i += 1
                third, odds = _num(cells[i][1]), _num(cells[i + 1][1])
                i += 2
                s = second[col]
            else:
                s, third, odds = _num(cells[i][1]), None, _num(cells[i + 1][1])
                i += 2
            if s is not None and odds is not None:
                out.append((col + 1, int(s), int(third) if third else None, odds))
    return out


def parse(html: str, page: str) -> dict[str, dict[str, float]]:
    """{券種: {組番: オッズ}}。組番は払戻と同じ書き方（2連複・3連複は小さい順）。"""
    bodies = [b for b in TABLE.findall(html) if "oddsPoint" in b]
    result = {}
    for ticket, body in zip(PAGES[page], bodies):
        three = ticket.startswith("3")
        combos = {}
        for a, b, c, o in _grid(body, three):
            boats = [a, b] + ([c] if three else [])
            if ticket.endswith("複"):
                boats = sorted(boats)
            combos["-".join(map(str, boats))] = o
        result[ticket] = combos
    return result


def fetch(d: date, venue: str, race: int, interval: float = 1.0, session: requests.Session | None = None) -> dict[str, dict[str, float]]:
    """3ページを順に取る（ページの間は少し空ける）。締切前は刻々と変わるので、見るたびに取り直す。"""
    s = session or requests.Session()
    out = {}
    for i, page in enumerate(PAGES):
        if i:
            time.sleep(interval)
        resp = s.get(
            URL.format(page=page), params={"rno": race, "jcd": venue, "hd": f"{d:%Y%m%d}"},
            headers={"User-Agent": "boatrace-yosou personal research"}, timeout=20,
        )
        resp.raise_for_status()
        out.update(parse(resp.text, page))
    return out


def collect(start: date, end: date, per_day: int, interval: float = 2.0, seed: int = 0) -> None:
    """検証用に、過去レースの（締切時点の）オッズを data/odds に保存する。1日 per_day レースを無作為に選ぶ。"""
    import json
    import random
    from datetime import timedelta

    from .download import DATA_DIR, download_day
    from .parse import parse_b

    out_dir = DATA_DIR.parent / "odds"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    s = requests.Session()
    d = start
    while d <= end:
        for kind in "BK":
            download_day(kind, d, session=s)
            time.sleep(interval)
        b = DATA_DIR / "B" / f"b{d:%y%m%d}.txt"
        races = sorted(set(map(tuple, parse_b(b.read_text(encoding="utf-8"), d)[["venue", "race"]].values))) if b.exists() else []
        for venue, race in rng.sample(races, min(per_day, len(races))):
            path = out_dir / f"{d:%y%m%d}_{venue}_{race:02d}.json"
            if path.exists():
                continue
            try:
                path.write_text(json.dumps(fetch(d, venue, int(race), interval, s), ensure_ascii=False))
                print(d, venue, race, "ok", flush=True)
            except Exception as e:
                print(d, venue, race, "失敗", e, flush=True)
            time.sleep(interval)
        d += timedelta(days=1)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="検証用に過去レースのオッズを集める")
    p.add_argument("start", type=date.fromisoformat)
    p.add_argument("end", type=date.fromisoformat)
    p.add_argument("--per-day", type=int, default=25)
    a = p.parse_args()
    collect(a.start, a.end, a.per_day)
