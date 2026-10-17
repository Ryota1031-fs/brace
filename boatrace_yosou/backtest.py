"""過去データで的中率・回収率を確かめる。

日付で前半を学習、後半を検証に分け、各レースで確率上位 N 点を100円ずつ買ったとして集計する。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .constants import TICKET_TYPES
from .model import RACE_KEY, GBMModel, RuleModel, add_features, top_tickets
from .parse import build_dataset

ROOT = Path(__file__).resolve().parent.parent


def merge_training(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    e = add_features(data["entries"])
    r = data["results"][RACE_KEY + ["boat", "position"]]
    df = e.merge(r, on=RACE_KEY + ["boat"], how="inner")
    # 6艇そろって着順が出ているレースだけを使う（欠場・F・不成立を除く）
    full = df.groupby(RACE_KEY)["position"].transform(lambda s: s.notna().sum() == 6)
    return df[full].reset_index(drop=True)


def evaluate(df: pd.DataFrame, probs: pd.Series, payouts: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.assign(prob=probs.values)
    # 同着だと同じ組番の払戻が複数行になることがあるので合計しておく
    pay = payouts.groupby(RACE_KEY + ["ticket", "combo"])["payout"].sum()
    stats = {t: {"races": 0, "hits": 0, "cost": 0, "return": 0} for t in TICKET_TYPES}
    for key, g in df.groupby(RACE_KEY):
        picks = top_tickets(dict(zip(g["boat"], g["prob"])), n)
        for t, combos in picks.items():
            s = stats[t]
            s["races"] += 1
            s["cost"] += 100 * len(combos)
            hit = False
            for combo, _ in combos:
                v = pay.get((*key, t, combo))
                if v is not None:
                    s["return"] += int(v)
                    hit = True
            s["hits"] += hit
    out = pd.DataFrame(stats).T
    out["的中率"] = (out["hits"] / out["races"]).round(3)
    out["回収率"] = (out["return"] / out["cost"]).round(3)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="ルールベースと LightGBM の的中率・回収率を比べる")
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--top", type=int, default=5, help="券種ごとに買う点数")
    a = p.parse_args()

    data = build_dataset(ROOT / "data" / "raw")
    df = merge_training(data)
    if df.empty:
        raise SystemExit("学習データがありません。先に download.py でデータを取得してください。")
    dates = sorted(df["date"].unique())
    cut = dates[int(len(dates) * (1 - a.test_ratio))]
    train, test = df[df["date"] < cut], df[df["date"] >= cut].reset_index(drop=True)
    print(f"学習 {train['date'].nunique()}日 / 検証 {test['date'].nunique()}日（{cut} 以降）, 検証レース数 {test.groupby(RACE_KEY).ngroups}")

    rule = RuleModel().fit(train)
    gbm = GBMModel().fit(train)
    gbm.save(ROOT / "models" / "gbm.txt")
    pd.Series(rule.lane_win).to_json(ROOT / "models" / "lane_win.json")

    for name, m in [("ルールベース", rule), ("LightGBM", gbm)]:
        print(f"\n== {name}（上位{a.top}点）==")
        print(evaluate(test, m.predict(test), data["payouts"], a.top)[["races", "的中率", "回収率"]])


if __name__ == "__main__":
    main()
