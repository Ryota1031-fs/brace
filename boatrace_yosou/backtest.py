"""過去データで的中率・回収率を確かめる。

日付で前半を学習、後半を検証に分け、各レースで確率上位 N 点を100円ずつ買ったとして集計する。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import TICKET_TYPES
from .history import HISTORY_FEATURES, add_history
from .model import FEATURES, RACE_KEY, GBMModel, RuleModel, add_features, top_tickets
from .parse import build_dataset

ROOT = Path(__file__).resolve().parent.parent


def merge_training(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    e = add_history(add_features(data["entries"]), data["results"])
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
    gbm_base = GBMModel(FEATURES).fit(train)
    gbm_hist = GBMModel(FEATURES + HISTORY_FEATURES).fit(train)

    print("ルールベースの重み:", {k: round(v, 3) for k, v in rule.weights.items()})
    for name, m in [("ルールベース", rule), ("LightGBM（出走表のみ）", gbm_base), ("LightGBM（出走表＋過去成績）", gbm_hist)]:
        print(f"\n== {name}（上位{a.top}点）==")
        probs = m.predict(test)
        won = (test["position"] == 1).to_numpy()
        # 実際の1着艇に付けた確率の対数の平均（0に近いほど当てはまりが良い）。点数の区切りに左右されにくい
        print(f"1着艇への当てはまり: {np.log(np.clip(probs.to_numpy()[won], 1e-9, 1)).mean():.4f}")
        print(evaluate(test, probs, data["payouts"], a.top)[["races", "的中率", "回収率"]])

    imp = pd.Series(gbm_hist.model.booster_.feature_importance("gain"), index=gbm_hist.features)
    print("\nLightGBM（出走表＋過去成績）で効いている項目（上位15）:")
    print((imp / imp.sum()).sort_values(ascending=False).head(15).round(3))

    # 画面で使うモデルは全期間で学び直して保存する
    models = ROOT / "models"
    GBMModel(FEATURES + HISTORY_FEATURES).fit(df).save(models / "gbm.txt")
    full_rule = RuleModel().fit(df)
    pd.Series(full_rule.lane_win).to_json(models / "lane_win.json")
    pd.Series(full_rule.weights).to_json(models / "rule_weights.json")
    # 画面で直近成績を計算するための過去成績（必要な列だけ）
    data["results"][["date", "venue", "race", "boat", "racer_no", "motor_no", "boat_no", "position", "st"]].to_csv(
        models / "results_history.csv.gz", index=False)


if __name__ == "__main__":
    main()
