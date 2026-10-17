"""競艇予想の画面（Streamlit）。  起動: streamlit run app.py"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from boatrace_yosou.constants import TICKET_TYPES, VENUES
from boatrace_yosou.download import DATA_DIR, download_day, extract_lzh
from boatrace_yosou.model import RACE_KEY, GBMModel, RuleModel, add_features, top_tickets
from boatrace_yosou.backtest import evaluate
from boatrace_yosou.history import add_history
from boatrace_yosou.parse import parse_b, parse_k

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="競艇予想", layout="wide")
st.title("競艇予想（試作）")

with st.sidebar:
    race_date = st.date_input("開催日", value=date.today())
    if st.button("この日の出走表を公式から取得"):
        try:
            path = download_day("B", race_date)
            st.success("取得しました" if path else "この日の出走表はまだありません")
        except Exception as e:  # 通信エラーなどはそのまま見せる
            st.error(f"取得できませんでした: {e}")
    uploaded = st.file_uploader("または番組表ファイル（.lzh / .txt）を読み込む", type=["lzh", "txt"])
    top_n = st.slider("券種ごとの表示点数", 1, 10, 5)
    refresh_results = st.button("この日の結果を公式から取り直す", help="当日の結果は夜にまとめて公開されます")


@st.cache_resource
def load_models():
    lane = None
    if (MODEL_DIR / "lane_win.json").exists():
        lane = {int(k): v for k, v in json.loads((MODEL_DIR / "lane_win.json").read_text()).items()}
    gbm = GBMModel.load(MODEL_DIR / "gbm.txt") if (MODEL_DIR / "gbm.txt").exists() else None
    return RuleModel(lane), gbm


@st.cache_data
def load_history(local_files: tuple[str, ...]) -> pd.DataFrame:
    """お渡しした過去成績と、このPCで取り込んだ結果ファイルをまとめる（直近成績の計算用）。"""
    parts = []
    if (MODEL_DIR / "results_history.csv.gz").exists():
        h = pd.read_csv(MODEL_DIR / "results_history.csv.gz", dtype={"venue": str})
        h["date"] = pd.to_datetime(h["date"]).dt.date
        parts.append(h)
    for f in local_files:
        p = Path(f)
        d = date(2000 + int(p.stem[1:3]), int(p.stem[3:5]), int(p.stem[5:7]))
        parts.append(parse_k(p.read_text(encoding="utf-8"), d)[0])
    if not parts:
        return pd.DataFrame()
    cols = ["date", "venue", "race", "boat", "racer_no", "position", "st"]
    return pd.concat([x[cols] for x in parts if not x.empty], ignore_index=True).drop_duplicates(["date", "venue", "race", "boat"])


if uploaded is not None:
    blob = uploaded.getvalue()
    text = extract_lzh(blob) if uploaded.name.endswith(".lzh") else blob.decode("cp932", errors="replace")
else:
    path = DATA_DIR / "B" / f"b{race_date:%y%m%d}.txt"
    tried = st.session_state.setdefault("auto_fetched", set())
    if not path.exists() and ("B", race_date) not in tried:
        # 選んだ日の出走表が手元になければ、公式から自動で取りにいく（同じ日は1回だけ）
        tried.add(("B", race_date))
        with st.spinner("出走表を公式から取得しています…"):
            try:
                path = download_day("B", race_date) or path
            except Exception as e:
                st.warning(f"出走表を自動で取得できませんでした: {e}")
    text = path.read_text(encoding="utf-8") if path.exists() else ""

entries = parse_b(text, race_date) if text else pd.DataFrame()
if entries.empty:
    st.info("この日の出走表はまだ公式に出ていません（たいてい前日の夕方以降に出ます）。あとで左のボタンを押すか、ファイルを読み込んでください。")
    st.stop()



def fetch_once(kind: str, d: date, force: bool = False) -> Path | None:
    """手元になければ公式から取る。同じ日・同じ種類は1回の起動で1回だけ（取り直しボタンを除く）。"""
    path = DATA_DIR / kind / f"{kind.lower()}{d:%y%m%d}.txt"
    tried = st.session_state.setdefault("auto_fetched", set())
    if (force or not path.exists()) and (force or (kind, d) not in tried) and d <= date.today():
        tried.add((kind, d))
        if force and path.exists():
            path.unlink()
        try:
            download_day(kind, d)
        except Exception as e:
            st.warning(f"公式から取得できませんでした: {e}")
    return path if path.exists() else None


k_path = fetch_once("K", race_date, force=refresh_results)
results, payouts = parse_k(k_path.read_text(encoding="utf-8"), race_date) if k_path else (pd.DataFrame(), pd.DataFrame())

df = add_features(entries)
local_k = tuple(sorted(str(p) for p in (DATA_DIR / "K").glob("k*.txt")))
df = add_history(df, load_history(local_k))
rule, gbm = load_models()
df["ルール"] = rule.predict(df).values
model_col = "ルール"
if gbm is not None:
    df["LightGBM"] = gbm.predict(df).values
    model_col = "LightGBM"
else:
    st.caption("学習済みモデルがないため、ルールベースの予想だけを表示しています（backtest を実行すると作られます）。")

venues = sorted(df["venue"].unique())
c1, c2, c3 = st.columns(3)
venue = c1.selectbox("場", venues, format_func=lambda v: f"{v} {VENUES.get(v, '')}")
races = sorted(df.loc[df["venue"] == venue, "race"].unique())
race = c2.selectbox("レース", races, format_func=lambda r: f"{r}R")
choices = ["ルール"] + (["LightGBM"] if gbm is not None else [])
model_col = c3.radio("予想に使うモデル", choices, index=len(choices) - 1, horizontal=True)

g = df[(df["venue"] == venue) & (df["race"] == race)].sort_values("boat")
show = g[["boat", "name", "class", "age", "nat_win", "loc_win", "motor_2ren", "boat_2ren"] + choices].rename(columns={
    "boat": "艇", "name": "選手", "class": "級", "age": "年齢", "nat_win": "全国勝率",
    "loc_win": "当地勝率", "motor_2ren": "モーター2連率", "boat_2ren": "ボート2連率",
})
st.subheader("各艇の1着確率")
fmt = {"全国勝率": "{:.2f}", "当地勝率": "{:.2f}", "モーター2連率": "{:.1f}", "ボート2連率": "{:.1f}"}
fmt.update({c: "{:.1%}" for c in choices})
st.dataframe(show.style.format(fmt), hide_index=True, width="stretch")

race_pay = payouts[(payouts["venue"] == venue) & (payouts["race"] == race)] if not payouts.empty else payouts
hit_combos = {(r.ticket, r.combo): r.payout for r in race_pay.itertuples()}

st.subheader(f"買い目の確率（{model_col}）")
picks = top_tickets(dict(zip(g["boat"], g[model_col])), top_n)
cols = st.columns(len(TICKET_TYPES))
for col, t in zip(cols, TICKET_TYPES):
    col.markdown(f"**{t}**")
    table = pd.DataFrame(picks[t], columns=["組番", "確率"])
    if hit_combos:
        table["結果"] = [f"的中 {hit_combos[(t, c)]:,}円" if (t, c) in hit_combos else "" for c in table["組番"]]
    col.dataframe(table.style.format({"確率": "{:.1%}"}), hide_index=True)

st.subheader("実際の結果")
race_res = results[(results["venue"] == venue) & (results["race"] == race)] if not results.empty else results
if race_res.empty:
    st.info("このレースの結果はまだ公式に出ていません。当日の結果は夜にまとめて公開されるので、あとで左の「結果を取り直す」を押してください。")
else:
    names = dict(zip(g["boat"], g["name"]))
    order = race_res.sort_values("position", na_position="last")
    st.dataframe(pd.DataFrame({
        "着": [str(int(p)) if pd.notna(p) else s for p, s in zip(order["position"], order["status"])],
        "艇": order["boat"],
        "選手": order["boat"].map(names),
        "進入": order["course"],
        "スタート": order["st"],
        "予想の1着確率": order["boat"].map(dict(zip(g["boat"], g[model_col]))),
    }).style.format({"予想の1着確率": "{:.1%}", "スタート": "{:.2f}", "進入": "{:.0f}"}), hide_index=True)
    pay_cols = st.columns(len(TICKET_TYPES))
    for col, t in zip(pay_cols, TICKET_TYPES):
        rows = race_pay[race_pay["ticket"] == t]
        col.markdown(f"**{t}**")
        col.markdown(f"### {' / '.join(rows['combo']) or 'なし'}")
        col.caption("払戻（100円あたり） " + " / ".join(f"{v:,}円" for v in rows["payout"]))

    # この日の全レースで、いまの設定（モデルと点数）どおりに100円ずつ買った場合の成績
    done = df.merge(payouts[RACE_KEY].drop_duplicates(), on=RACE_KEY)
    if not done.empty:
        st.subheader(f"この日の成績（{model_col}・券種ごとに上位{top_n}点）")
        summary = evaluate(done.reset_index(drop=True), done[model_col], payouts, top_n)
        summary = summary.rename(columns={"races": "レース数", "hits": "的中数", "cost": "購入額", "return": "払戻額"})
        st.dataframe(summary[["レース数", "的中数", "的中率", "購入額", "払戻額", "回収率"]].style.format(
            {"的中率": "{:.1%}", "回収率": "{:.1%}", "購入額": "{:,}円", "払戻額": "{:,}円"}), width="stretch")

st.caption("確率は出走表の数値だけから計算した目安です。展示タイムや直前のオッズは反映していません。")
