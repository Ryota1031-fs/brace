"""公式の番組表(B)・競走成績(K)ファイルをダウンロードして展開する。

公式サイトへの負荷を避けるため、1ファイルごとに間隔を空け、取得済みの日は飛ばす。
"""
from __future__ import annotations

import argparse
import io
import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE_URL = "https://www1.mbrace.or.jp/od2"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def file_url(kind: str, d: date) -> str:
    """kind は "B"（番組表）か "K"（競走成績）。"""
    k = kind.upper()
    return f"{BASE_URL}/{k}/{d:%Y%m}/{k.lower()}{d:%y%m%d}.lzh"


def extract_lzh(blob: bytes) -> str:
    import lhafile

    archive = lhafile.Lhafile(io.BytesIO(blob))
    texts = [archive.read(info.filename).decode("cp932", errors="replace") for info in archive.infolist()]
    return "\n".join(texts)


def download_day(kind: str, d: date, out_dir: Path = DATA_DIR, session: requests.Session | None = None) -> Path | None:
    out = out_dir / kind.upper() / f"{kind.lower()}{d:%y%m%d}.txt"
    if out.exists():
        return out
    s = session or requests.Session()
    resp = s.get(file_url(kind, d), timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(extract_lzh(resp.content), encoding="utf-8")
    return out


def download_range(start: date, end: date, kinds=("B", "K"), interval: float = 2.0) -> None:
    s = requests.Session()
    s.headers["User-Agent"] = "boatrace-yosou personal research"
    d = start
    while d <= end:
        for kind in kinds:
            target = DATA_DIR / kind / f"{kind.lower()}{d:%y%m%d}.txt"
            if target.exists():
                continue
            path = download_day(kind, d, session=s)
            print(f"{d} {kind}: {'ok' if path else 'なし'}")
            time.sleep(interval)
        d += timedelta(days=1)


def main() -> None:
    p = argparse.ArgumentParser(description="公式の番組表・競走成績をダウンロード")
    p.add_argument("start", type=date.fromisoformat)
    p.add_argument("end", type=date.fromisoformat)
    p.add_argument("--kinds", default="BK")
    p.add_argument("--interval", type=float, default=2.0, help="1ファイルごとの待ち秒数")
    a = p.parse_args()
    download_range(a.start, a.end, tuple(a.kinds), a.interval)


if __name__ == "__main__":
    main()
