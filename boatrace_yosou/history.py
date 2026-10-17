"""過去の競走成績から、選手ごとの直近成績を特徴量にする。

レース当日より前の結果だけを使う（当日以降の結果が混ざると検証の成績が実力以上に良く見えるため）。
- 直近180日: 出走数、1着率、2連対率、平均スタート
- 直近365日の枠別: その枠での出走数と1着率
率は出走数が少ないと振れるので、全体平均に向けてならしている。
"""
from __future__ import annotations

import pandas as pd

RECENT_DAYS = 180
LANE_DAYS = 365
PRIOR_N = 10  # この出走数ぶん全体平均を混ぜてならす

HISTORY_FEATURES = [
    "hist_starts", "hist_win", "hist_top2", "hist_st", "lane_starts", "lane_win",
    "hist_win_diff", "hist_st_diff", "lane_win_diff",
]


def _daily(results: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    r = results.assign(
        date=pd.to_datetime(results["date"]),
        start=1,
        win=(results["position"] == 1).astype(int),
        top2=(results["position"] <= 2).astype(int),
        st_sum=results["st"].where(results["st"].between(0, 1)).fillna(0),
        st_n=results["st"].between(0, 1).astype(int),
    )
    return r.groupby(by + ["date"], as_index=False)[["start", "win", "top2", "st_sum", "st_n"]].sum()


def _rolling_before(daily: pd.DataFrame, by: list[str], days: int) -> pd.DataFrame:
    """各 (by, 日付) について、その日より前 days 日間の合計を返す。"""
    cols = ["start", "win", "top2", "st_sum", "st_n"]
    daily = daily.sort_values(by + ["date"])
    rolled = (
        daily.set_index("date").groupby(by)[cols]
        .rolling(f"{days}D", closed="left").sum()
        .fillna(0).reset_index()
    )
    return rolled


def add_history(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """df（出走表）に直近成績の列を足す。results は過去の競走成績（parse_k の着順表）。"""
    out = df.copy()
    if results is None or results.empty:
        for c in HISTORY_FEATURES:
            out[c] = float("nan")
        return out
    res = results[results["racer_no"].notna()]
    key_date = pd.to_datetime(out["date"])
    base_win = (res["position"] == 1).mean()
    base_top2 = (res["position"] <= 2).mean()
    base_st = res["st"].where(res["st"].between(0, 1)).mean()
    lane_base = (res["position"] == 1).groupby(res["boat"]).mean()

    # 出走表の日付ごとに「その日より前」の集計を引けるよう、出走表の日付も行として足してから集計する
    def stats_for(by: list[str], days: int, left_on: list[str]) -> pd.DataFrame:
        daily = _daily(res, by)
        targets = out[left_on].assign(date=key_date).rename(columns=dict(zip(left_on, by))).drop_duplicates()
        targets = targets.assign(start=0, win=0, top2=0, st_sum=0.0, st_n=0)
        merged = pd.concat([daily, targets], ignore_index=True).groupby(by + ["date"], as_index=False).sum()
        rolled = _rolling_before(merged, by, days)
        return targets[by + ["date"]].merge(rolled, on=by + ["date"], how="left")

    s = stats_for(["racer_no"], RECENT_DAYS, ["racer_no"])
    s["hist_starts"] = s["start"]
    s["hist_win"] = (s["win"] + PRIOR_N * base_win) / (s["start"] + PRIOR_N)
    s["hist_top2"] = (s["top2"] + PRIOR_N * base_top2) / (s["start"] + PRIOR_N)
    s["hist_st"] = (s["st_sum"] + PRIOR_N * base_st) / (s["st_n"] + PRIOR_N)
    out = out.assign(date_ts=key_date).merge(
        s[["racer_no", "date", "hist_starts", "hist_win", "hist_top2", "hist_st"]].rename(columns={"date": "date_ts"}),
        on=["racer_no", "date_ts"], how="left",
    )

    l = stats_for(["racer_no", "boat"], LANE_DAYS, ["racer_no", "boat"])
    prior = l["boat"].map(lane_base).fillna(base_win)
    l["lane_starts"] = l["start"]
    l["lane_win"] = (l["win"] + PRIOR_N * prior) / (l["start"] + PRIOR_N)
    out = out.merge(
        l[["racer_no", "boat", "date", "lane_starts", "lane_win"]].rename(columns={"date": "date_ts"}),
        on=["racer_no", "boat", "date_ts"], how="left",
    )
    out = out.drop(columns="date_ts")
    # 同じレースの他の艇と比べた差（強さはレース内の相対で効くため）
    g = out.groupby(["date", "venue", "race"])
    for c in ["hist_win", "hist_st", "lane_win"]:
        out[f"{c}_diff"] = out[c] - g[c].transform("mean")
    return out
