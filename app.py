"""競艇予想の画面（Streamlit）。  起動: streamlit run app.py"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from boatrace_yosou.constants import TICKET_TYPES, VENUES
from boatrace_yosou.download import DATA_DIR, download_day, extract_lzh
from boatrace_yosou.model import RACE_KEY, GBMModel, RuleModel, add_exhibition, add_features, top_tickets
from boatrace_yosou import beforeinfo
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
    st.divider()
    if st.button("アプリを最新版に更新"):
        from boatrace_yosou.update import update

        try:
            files = update()
            if files:
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success(f"{len(files)}個のファイルを更新しました。黒い画面を閉じて、もう一度 start で開き直してください。")
            else:
                st.info("すでに最新です。")
        except Exception as e:
            st.error(f"更新できませんでした: {e}")


@st.cache_resource
def load_models():
    lane = None
    if (MODEL_DIR / "lane_win.json").exists():
        lane = {int(k): v for k, v in json.loads((MODEL_DIR / "lane_win.json").read_text()).items()}
    weights = json.loads((MODEL_DIR / "rule_weights.json").read_text()) if (MODEL_DIR / "rule_weights.json").exists() else None
    gbm = GBMModel.load(MODEL_DIR / "gbm.txt") if (MODEL_DIR / "gbm.txt").exists() else None
    gbm_ex = GBMModel.load(MODEL_DIR / "gbm_ex.txt") if (MODEL_DIR / "gbm_ex.txt").exists() else None
    return RuleModel(lane, weights), gbm, gbm_ex


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
    cols = ["date", "venue", "race", "boat", "racer_no", "motor_no", "boat_no", "position", "st"]
    return pd.concat([x.reindex(columns=cols) for x in parts if not x.empty], ignore_index=True).drop_duplicates(["date", "venue", "race", "boat"])


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
    # 当日の結果は場ごとに夜までかけて増えるので、30分たったら取り直す
    stale = d == date.today() and path.exists() and time.time() - path.stat().st_mtime > 1800
    if (force or stale or not path.exists()) and (force or stale or (kind, d) not in tried) and d <= date.today():
        tried.add((kind, d))
        if (force or stale) and path.exists():
            path.unlink()
        try:
            download_day(kind, d)
        except Exception as e:
            st.warning(f"公式から取得できませんでした: {e}")
    return path if path.exists() else None


k_path = fetch_once("K", race_date, force=refresh_results)
results, payouts = parse_k(k_path.read_text(encoding="utf-8"), race_date) if k_path else (pd.DataFrame(), pd.DataFrame())

rule, gbm, gbm_ex = load_models()
venues = sorted(entries["venue"].unique())
c1, c2, c3 = st.columns(3)
venue = c1.selectbox("場", venues, format_func=lambda v: f"{v} {VENUES.get(v, '')}")
races = sorted(entries.loc[entries["venue"] == venue, "race"].unique())
race = c2.selectbox("レース", races, format_func=lambda r: f"{r}R")
choices = ["ルール"] + (["LightGBM"] if gbm is not None else [])
model_col = c3.radio("予想に使うモデル", choices, index=len(choices) - 1, horizontal=True)

# 展示タイム: 結果ファイルがある日はそこから、まだの日は直前情報ページから（このレースだけ取りにいく）
ex_store = st.session_state.setdefault("exhibition", {})
key = (race_date, venue, race)
ex_cols = RACE_KEY + ["boat", "exhibition"]
from_results = results[ex_cols] if not results.empty else pd.DataFrame(columns=ex_cols)
race_done = ((from_results["venue"] == venue) & (from_results["race"] == race)).any()
if not race_done and race_date == date.today():
    refetch = st.button("展示タイムを取り直す（直前情報）", help="展示航走はレースのおよそ20〜30分前。終わると公式に出ます")
    if refetch or key not in ex_store:
        try:
            ex_store[key] = beforeinfo.fetch(race_date, venue, race)
        except Exception as e:
            st.warning(f"直前情報を取得できませんでした: {e}")
from_before = [v.assign(date=k[0], venue=k[1], race=k[2]) for k, v in ex_store.items() if k[0] == race_date]
# 同じレースが両方にあれば結果ファイルの方を使う
ex_all = pd.concat([from_results] + [x[ex_cols] for x in from_before], ignore_index=True).drop_duplicates(
    RACE_KEY + ["boat"], keep="first")

df = add_features(entries)
local_k = tuple(sorted(str(p) for p in (DATA_DIR / "K").glob("k*.txt")))
df = add_history(df, load_history(local_k))
ex_all = ex_all[RACE_KEY + ["boat", "exhibition"]].astype({"race": int, "boat": int, "exhibition": float})
df = add_exhibition(df.drop(columns="exhibition").merge(ex_all, on=RACE_KEY + ["boat"], how="left"))
# 6艇とも展示タイムがそろったレースだけ「展示あり」で予想する
has_ex = df.groupby(RACE_KEY)["ex_diff"].transform(lambda x: x.notna().all())
df["ルール"] = rule.predict(df).values
if gbm is not None:
    p = gbm.predict(df)
    if gbm_ex is not None and has_ex.any():
        p = p.where(~has_ex, gbm_ex.predict(df))
    df["LightGBM"] = p.values
else:
    st.caption("学習済みモデルがないため、ルールベースの予想だけを表示しています（backtest を実行すると作られます）。")

g = df[(df["venue"] == venue) & (df["race"] == race)].sort_values("boat")
if has_ex.loc[g.index].all():
    st.success("このレースは展示タイムを反映した予想です。")
else:
    st.caption("このレースはまだ展示タイムが出ていないため、出走表だけで予想しています。展示のあとに「展示タイムを取り直す」を押すと反映されます。")
# 最近90日のモーター・ボートの2連対率（過去の結果が手元にあるときだけ出る）
recent = [c for c in ["motor_recent_top2", "boat_recent_top2"] if g[c].nunique() > 1]
show = g[["boat", "name", "class", "age", "nat_win", "loc_win", "motor_2ren", "boat_2ren"] + recent + ["exhibition"] + choices].rename(columns={
    "exhibition": "展示タイム",
    "boat": "艇", "name": "選手", "class": "級", "age": "年齢", "nat_win": "全国勝率",
    "loc_win": "当地勝率", "motor_2ren": "モーター2連率", "boat_2ren": "ボート2連率",
    "motor_recent_top2": "モーター近況", "boat_recent_top2": "ボート近況",
})
st.subheader("各艇の1着確率")
fmt = {"全国勝率": "{:.2f}", "当地勝率": "{:.2f}", "モーター2連率": "{:.1f}", "ボート2連率": "{:.1f}", "展示タイム": "{:.2f}"}
fmt.update({c: "{:.1%}" for c in choices + ["モーター近況", "ボート近況"] if c in show})
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

st.caption("確率は出走表と展示タイムから計算した目安です。直前のオッズは反映していません。")
