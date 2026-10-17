"""番組表(B)・競走成績(K)テキストを表に変換する。

どちらのファイルも場ごとに "01BBGN"〜"01BEND"（K は "01KBGN"〜"01KEND"）で区切られている。
レース見出しは "Ｈ１８００ｍ" のような距離表記を含む行で判定する（K の冒頭にある払戻一覧と区別するため）。
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

_NORMALIZE = str.maketrans("０１２３４５６７８９．－ＲＨｍ", "0123456789.-RHm")

BLOCK_START = re.compile(r"^(\d{2})([BK])BGN")
BLOCK_END = re.compile(r"^(\d{2})([BK])END")
RACE_HEADER = re.compile(r"^\s*(\d{1,2})R\s.*H\s*\d{3,4}\s*m")

# モーター・ボート番号が3桁だと前の数字と空白なしで続くことがある（例: "29.55135 37.80"）
# 例: "1 4320峰　　竜太36佐賀52A1 7.89 57.47 8.12 62.50 33 38.52 12 39.33 ..."
B_ENTRY = re.compile(
    r"^([1-6]) (\d{4})(\D+?)(\d{2})(\D+?)(\d{2})([AB][12])\s+"
    r"(\d+\.\d{2})\s*(\d+\.\d{2})\s*(\d+\.\d{2})\s*(\d+\.\d{2})\s*"
    r"(\d+)\s*(\d+\.\d{2})\s*(\d+)\s*(\d+\.\d{2})"
)

# 例: "  01  1 4320 峰　　竜太　　　　12   34  6.78   1    0.12     1.49.8"
K_ENTRY = re.compile(r"^\s*([0-9FLKS]{1,2}\S?)\s+([1-6])\s+(\d{4})\s+(.*)$")
K_PAYOUT = re.compile(r"(2連単|2連複|3連単|3連複)\s+(\d(?:-\d){1,2})\s+(\d+)(?:\s+人気\s+(\d+))?")
K_WIND = re.compile(r"風\s*(\S+?)\s+(\d+)m")
K_WAVE = re.compile(r"波\s*(\d+)cm")


def _norm(line: str) -> str:
    return line.translate(_NORMALIZE).rstrip("\r\n")


def _blocks(text: str):
    """場ごとに (場コード, 行のリスト) を返す。"""
    venue, lines = None, []
    for raw in text.splitlines():
        line = _norm(raw)
        if m := BLOCK_START.match(line):
            venue, lines = m.group(1), []
        elif BLOCK_END.match(line):
            if venue:
                yield venue, lines
            venue = None
        elif venue:
            lines.append(line)


def _date_from_name(path: Path) -> date:
    return datetime.strptime(re.search(r"(\d{6})", path.stem).group(1), "%y%m%d").date()


def parse_b(text: str, race_date: date) -> pd.DataFrame:
    rows = []
    for venue, lines in _blocks(text):
        race_no = None
        for line in lines:
            if m := RACE_HEADER.match(line):
                race_no = int(m.group(1))
                continue
            if race_no is None:
                continue
            if m := B_ENTRY.match(line):
                g = m.groups()
                rows.append({
                    "date": race_date, "venue": venue, "race": race_no,
                    "boat": int(g[0]), "racer_no": int(g[1]),
                    "name": g[2].replace("　", "").strip(),
                    "age": int(g[3]), "branch": g[4].replace("　", "").strip(),
                    "weight": int(g[5]), "class": g[6],
                    "nat_win": float(g[7]), "nat_2ren": float(g[8]),
                    "loc_win": float(g[9]), "loc_2ren": float(g[10]),
                    "motor_no": int(g[11]), "motor_2ren": float(g[12]),
                    "boat_no": int(g[13]), "boat_2ren": float(g[14]),
                })
    return pd.DataFrame(rows)


def _to_float(tok: str):
    try:
        return float(tok)
    except ValueError:
        return None


def parse_k(text: str, race_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(着順の表, 払戻の表) を返す。着順が数字でない艇（F・L・欠場など）は position を空にする。"""
    results, payouts = [], []
    for venue, lines in _blocks(text):
        race_no, wind, wave = None, None, None
        for line in lines:
            if m := RACE_HEADER.match(line):
                race_no = int(m.group(1))
                w = K_WIND.search(line)
                v = K_WAVE.search(line)
                wind = int(w.group(2)) if w else None
                wave = int(v.group(1)) if v else None
                continue
            if race_no is None:
                continue
            if m := K_PAYOUT.search(line):
                payouts.append({
                    "date": race_date, "venue": venue, "race": race_no,
                    "ticket": m.group(1), "combo": m.group(2),
                    "payout": int(m.group(3)),
                    "popularity": int(m.group(4)) if m.group(4) else None,
                })
                continue
            if m := K_ENTRY.match(line):
                pos_tok, boat, racer_no, rest = m.groups()
                nums = [t for t in rest.split() if t.isascii()]
                results.append({
                    "date": race_date, "venue": venue, "race": race_no,
                    "boat": int(boat), "racer_no": int(racer_no),
                    "position": int(pos_tok) if pos_tok.isdigit() else None,
                    "status": None if pos_tok.isdigit() else pos_tok,
                    "motor_no": int(nums[0]) if nums and nums[0].isdigit() else None,
                    "boat_no": int(nums[1]) if len(nums) > 1 and nums[1].isdigit() else None,
                    "exhibition": _to_float(nums[2]) if len(nums) > 2 else None,
                    "course": int(nums[3]) if len(nums) > 3 and nums[3].isdigit() else None,
                    "st": _to_float(nums[4]) if len(nums) > 4 else None,
                    "wind": wind, "wave": wave,
                })
    return pd.DataFrame(results), pd.DataFrame(payouts)


def build_dataset(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """data/raw の B・K テキストをまとめて読み込む。"""
    entries = [parse_b(p.read_text(encoding="utf-8"), _date_from_name(p)) for p in sorted((raw_dir / "B").glob("b*.txt"))]
    res, pay = [], []
    for p in sorted((raw_dir / "K").glob("k*.txt")):
        r, y = parse_k(p.read_text(encoding="utf-8"), _date_from_name(p))
        res.append(r)
        pay.append(y)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return {"entries": cat(entries), "results": cat(res), "payouts": cat(pay)}
