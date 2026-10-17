"""各艇の1着確率を出し、そこから2連単・2連複・3連単・3連複の確率を計算する。

- ルールベース: 枠ごとの基本勝率 × 勝率・モーター・級別の補正
- LightGBM: 出走表の数値から1着かどうかを学習し、レース内で合計1になるよう正規化
買い目の確率は Harville 法（1着確率から順に条件付きで掛け合わせる）で出す。
"""
from __future__ import annotations

from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CLASS_RANK

RACE_KEY = ["date", "venue", "race"]

FEATURES = [
    "boat", "class_rank", "age", "weight", "nat_win", "nat_2ren", "loc_win", "loc_2ren",
    "motor_2ren", "boat_2ren", "venue_code", "race",
    "nat_win_diff", "nat_win_rank", "motor_2ren_diff", "class_rank_diff",
]

# 学習データがないときに使う枠別の1着率の目安（全国の傾向をおおまかに置いた値）
DEFAULT_LANE_WIN = {1: 0.55, 2: 0.14, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.03}


def add_features(entries: pd.DataFrame) -> pd.DataFrame:
    df = entries.copy()
    df["class_rank"] = df["class"].map(CLASS_RANK).fillna(1)
    df["venue_code"] = df["venue"].astype(int)
    g = df.groupby(RACE_KEY)
    df["nat_win_diff"] = df["nat_win"] - g["nat_win"].transform("mean")
    df["nat_win_rank"] = g["nat_win"].rank(ascending=False, method="min")
    df["motor_2ren_diff"] = df["motor_2ren"] - g["motor_2ren"].transform("mean")
    df["class_rank_diff"] = df["class_rank"] - g["class_rank"].transform("mean")
    return df


def normalize_in_race(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] / df.groupby(RACE_KEY)[col].transform("sum")


class RuleModel:
    def __init__(self, lane_win: dict[int, float] | None = None):
        self.lane_win = lane_win or DEFAULT_LANE_WIN

    def fit(self, df: pd.DataFrame) -> "RuleModel":
        """学習データがあれば枠別の1着率を実データから置き換える。"""
        rate = (df["position"] == 1).groupby(df["boat"]).mean()
        if len(rate) == 6:
            self.lane_win = rate.to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        base = df["boat"].map(self.lane_win)
        strength = (
            base
            * np.exp(0.35 * df["nat_win_diff"])
            * np.exp(0.02 * df["motor_2ren_diff"])
            * np.exp(0.15 * df["class_rank_diff"])
        )
        tmp = df[RACE_KEY].assign(s=strength.values)
        return normalize_in_race(tmp, "s")


class GBMModel:
    def __init__(self, features: list[str] | None = None):
        self.model = None
        self.features = features or FEATURES

    def fit(self, df: pd.DataFrame) -> "GBMModel":
        import lightgbm as lgb

        y = (df["position"] == 1).astype(int)
        self.model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, verbose=-1,
        )
        self.model.fit(df[self.features], y, categorical_feature=["venue_code"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        p = self.model.predict_proba(df[self.features])[:, 1]
        tmp = df[RACE_KEY].assign(p=p)
        return normalize_in_race(tmp, "p")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))

    @classmethod
    def load(cls, path: Path) -> "GBMModel":
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(path))
        m = cls(booster.feature_name())

        class _Wrap:
            def predict_proba(self, X):
                p = booster.predict(X)
                return np.column_stack([1 - p, p])

        m.model = _Wrap()
        return m


def ticket_probs(win_prob: dict[int, float]) -> dict[str, dict[str, float]]:
    """艇番→1着確率 から、券種ごとの {組番: 確率} を返す。"""
    boats = sorted(win_prob)
    p = win_prob
    exacta = {}
    trifecta = {}
    for i, j in permutations(boats, 2):
        exacta[f"{i}-{j}"] = p[i] * p[j] / max(1 - p[i], 1e-9)
    for i, j, k in permutations(boats, 3):
        trifecta[f"{i}-{j}-{k}"] = (
            p[i] * p[j] / max(1 - p[i], 1e-9) * p[k] / max(1 - p[i] - p[j], 1e-9)
        )
    quinella = {f"{i}-{j}": exacta[f"{i}-{j}"] + exacta[f"{j}-{i}"] for i, j in combinations(boats, 2)}
    trio = {
        f"{i}-{j}-{k}": sum(trifecta["-".join(map(str, o))] for o in permutations((i, j, k)))
        for i, j, k in combinations(boats, 3)
    }
    return {"2連単": exacta, "2連複": quinella, "3連単": trifecta, "3連複": trio}


def top_tickets(win_prob: dict[int, float], n: int = 5) -> dict[str, list[tuple[str, float]]]:
    return {
        t: sorted(probs.items(), key=lambda kv: -kv[1])[:n]
        for t, probs in ticket_probs(win_prob).items()
    }
