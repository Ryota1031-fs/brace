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
from boatrace_yosou.parse import parse_b

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


@st.cache_resource
def load_models():
    lane = None
    if (MODEL_DIR / "lane_win.json").exists():
        lane = {int(k): v for k, v in json.loads((MODEL_DIR / "lane_win.json").read_text()).items()}
    gbm = GBMModel.load(MODEL_DIR / "gbm.txt") if (MODEL_DIR / "gbm.txt").exists() else None
    return RuleModel(lane), gbm


if uploaded is not None:
    blob = uploaded.getvalue()
    text = extract_lzh(blob) if uploaded.name.endswith(".lzh") else blob.decode("cp932", errors="replace")
else:
    path = DATA_DIR / "B" / f"b{race_date:%y%m%d}.txt"
    text = path.read_text(encoding="utf-8") if path.exists() else ""

entries = parse_b(text, race_date) if text else pd.DataFrame()
if entries.empty:
    st.info("出走表がありません。左のボタンで取得するか、ファイルを読み込んでください。")
    st.stop()

df = add_features(entries)
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

st.subheader(f"買い目の確率（{model_col}）")
picks = top_tickets(dict(zip(g["boat"], g[model_col])), top_n)
cols = st.columns(len(TICKET_TYPES))
for col, t in zip(cols, TICKET_TYPES):
    col.markdown(f"**{t}**")
    col.dataframe(pd.DataFrame(picks[t], columns=["組番", "確率"]).style.format({"確率": "{:.1%}"}), hide_index=True)

st.caption("確率は出走表の数値だけから計算した目安です。展示タイムや直前のオッズは反映していません。")
