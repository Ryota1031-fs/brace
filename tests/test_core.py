"""最小限の動作確認。

SAMPLE_B / SAMPLE_K は公式ファイルのレイアウトをまねて手で書いた形式見本で、実在のレース結果ではない。
実ファイルを取得したら、実ファイルでも読めることを確認すること。
"""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from boatrace_yosou.backtest import evaluate, merge_training
from boatrace_yosou.history import HISTORY_FEATURES, add_history
from boatrace_yosou.model import FEATURES, GBMModel, RuleModel, add_features, ticket_probs
from boatrace_yosou.parse import parse_b, parse_k

D = date(2026, 1, 1)

SAMPLE_B = """STARTB
01BBGN
ボートレース桐生
　１Ｒ  予選　　　　　 Ｈ１８００ｍ  電話投票締切予定１０：３５
-------------------------------------------------------------------------------
1 1001見本　太郎36群馬52A1 7.89 57.47 8.12 62.50 33 38.52 12 39.33 1 1 2 3
2 1002見本　次郎28東京54B1 5.10 30.00 4.80 28.00 21 31.00 44 30.50
3 1003見本　三郎41埼玉51A2 6.20 45.00 6.00 40.00 55 42.10 18 35.00
4 1004見本　四郎33静岡53B1 4.90 28.00 5.20 30.00 17 29.80 61 33.20
5 1005見本　五郎25福岡50B2 3.10 15.00 2.90 12.00 40 33.30 27 31.10
6 1006見本　六郎47大阪55B1 5.40 35.00 5.00 33.00 62 35.70 39 32.40
01BEND
FINALB
"""

SAMPLE_K = """STARTK
01KBGN
ボートレース桐生
   1R       予選                 H1800m  晴　  風  北　　 1m  波　  1cm
  着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸﾞ ﾚｰｽﾀｲﾑ
-------------------------------------------------------------------------------
  01  1 1001 見本　太郎　　　　33   12  6.78   1    0.12     1.49.8
  02  3 1003 見本　三郎　　　　55   18  6.80   3    0.15     1.51.2
  03  2 1002 見本　次郎　　　　21   44  6.85   2    0.18     1.52.0
  04  4 1004 見本　四郎　　　　17   61  6.90   4    0.20     1.53.1
  05  6 1006 見本　六郎　　　　62   39  6.82   6    0.17     1.54.0
  F   5 1005 見本　五郎　　　　40   27  6.95   5   F0.01      .  .
        ２連単   1-3        560  人気     2
        ２連複   1-3        380  人気     1
        ３連単   1-3-2     1230  人気     3
        ３連複   1-2-3      450  人気     2
01KEND
FINALK
"""


def test_parse_b():
    df = parse_b(SAMPLE_B, D)
    assert len(df) == 6
    r = df.iloc[0]
    assert (r.venue, r.race, r.boat, r.racer_no, r["name"], r["class"]) == ("01", 1, 1, 1001, "見本太郎", "A1")
    assert r.nat_win == 7.89 and r.motor_no == 33 and r.boat_2ren == 39.33


def test_parse_k():
    res, pay = parse_k(SAMPLE_K, D)
    assert len(res) == 6
    assert res.loc[res.boat == 3, "position"].item() == 2
    assert pd.isna(res.loc[res.boat == 5, "position"].item())
    assert res.loc[res.boat == 1, "course"].item() == 1 and res.loc[res.boat == 1, "st"].item() == 0.12
    assert set(pay.ticket) == {"2連単", "2連複", "3連単", "3連複"}
    assert pay.loc[pay.ticket == "3連単", "payout"].item() == 1230


def test_ticket_probs_sum_to_one():
    p = {1: 0.5, 2: 0.15, 3: 0.12, 4: 0.1, 5: 0.08, 6: 0.05}
    t = ticket_probs(p)
    for name, n in [("2連単", 30), ("2連複", 15), ("3連単", 120), ("3連複", 20)]:
        assert len(t[name]) == n
        assert sum(t[name].values()) == pytest.approx(1.0)


def _random_data(days=30, seed=0):
    """パイプラインが通ることだけを見るための乱数データ。"""
    rng = np.random.default_rng(seed)
    ents, ress, pays = [], [], []
    for d in pd.date_range("2026-01-01", periods=days).date:
        for race in range(1, 13):
            win = rng.uniform(3, 8, 6)
            strength = np.array([5, 1.5, 1.2, 1, 0.6, 0.3]) * np.exp(0.4 * (win - win.mean()))
            order = rng.choice(6, 6, replace=False, p=strength / strength.sum()) + 1
            for b in range(1, 7):
                ents.append(dict(date=d, venue="01", race=race, boat=b, racer_no=1000 + b, name="x", age=30,
                                 branch="x", weight=52, **{"class": "B1"}, nat_win=win[b - 1], nat_2ren=30.0,
                                 loc_win=win[b - 1], loc_2ren=30.0, motor_no=1, motor_2ren=rng.uniform(20, 50),
                                 boat_no=1, boat_2ren=30.0))
                ress.append(dict(date=d, venue="01", race=race, boat=b, racer_no=1000 + b, st=0.15,
                                 position=int(np.where(order == b)[0][0] + 1)))
            o = list(map(str, order))
            pays += [dict(date=d, venue="01", race=race, ticket="2連単", combo="-".join(o[:2]), payout=800),
                     dict(date=d, venue="01", race=race, ticket="3連単", combo="-".join(o[:3]), payout=5000)]
    return {"entries": pd.DataFrame(ents), "results": pd.DataFrame(ress), "payouts": pd.DataFrame(pays)}


def test_pipeline_runs():
    data = _random_data()
    df = merge_training(data)
    train, test = df[df.date < df.date.max()], df[df.date == df.date.max()].reset_index(drop=True)
    for m in (RuleModel().fit(train), GBMModel().fit(train), GBMModel(FEATURES + HISTORY_FEATURES).fit(train)):
        probs = m.predict(test)
        assert probs.groupby([test.date, test.venue, test.race]).sum().round(6).eq(1).all()
        out = evaluate(test, probs, data["payouts"], 5)
        assert out.loc["3連単", "races"] == 12


def test_history_uses_only_earlier_days():
    data = _random_data(days=5)
    h = add_history(data["entries"], data["results"])
    first = h[h.date == h.date.min()]
    assert (first.hist_starts == 0).all()
    # 3日目の時点では、その艇の選手は前の2日で12レースずつ走っている
    third = h[h.date == sorted(h.date.unique())[2]]
    assert (third.hist_starts == 24).all() and (third.lane_starts == 24).all()
