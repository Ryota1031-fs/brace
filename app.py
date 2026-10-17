"""競艇予想の画面（Streamlit）。  起動: streamlit run app.py"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from boatrace_yosou import beforeinfo, odds, ui
from boatrace_yosou.backtest import evaluate
from boatrace_yosou.constants import TICKET_TYPES, VENUES
from boatrace_yosou.download import DATA_DIR, download_day, extract_lzh
from boatrace_yosou.history import add_history
from boatrace_yosou.model import RACE_KEY, GBMModel, RuleModel, add_exhibition, add_features, ticket_probs
from boatrace_yosou.parse import parse_b, parse_k

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="競艇予想", page_icon="🚤", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🚤 設定")
    race_date = st.date_input("開催日", value=date.today())
    top_n = st.slider("券種ごとの表示点数", 1, 10, 5)
    st.markdown("**妙味（期待値）の目安**")
    ev_line = st.slider("期待値がこれ以上なら印を付ける", 1.0, 2.0, 1.2, 0.05,
                        help="期待値 = 当たる確率 × オッズ。1を超えると、確率のわりに配当が高い買い目です")
    min_prob = st.slider("ただし当たる確率がこれ以上のものだけ", 0.0, 0.10, 0.01, 0.005, format="%.3f",
                         help="確率がとても低い大穴は、予想の確率そのものがぶれやすいので外します")
    st.divider()
    refresh_results = st.button("この日の結果を取り直す", help="当日の結果は場ごとに、全レース終了後に公開されます", width="stretch")
    if st.button("この日の出走表を取り直す", width="stretch"):
        try:
            p = DATA_DIR / "B" / f"b{race_date:%y%m%d}.txt"
            p.unlink(missing_ok=True)
            st.success("取得しました" if download_day("B", race_date) else "この日の出走表はまだありません")
        except Exception as e:  # 通信エラーなどはそのまま見せる
            st.error(f"取得できませんでした: {e}")
    with st.expander("ファイルから読み込む"):
        uploaded = st.file_uploader("番組表ファイル（.lzh / .txt）", type=["lzh", "txt"])
    st.divider()
    if st.button("アプリを最新版に更新", type="primary", width="stretch"):
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


def fetch_once(kind: str, d: date, force: bool = False) -> Path | None:
    """手元になければ公式から取る。同じ日・同じ種類は1回の起動で1回だけ（取り直しボタンを除く）。"""
    path = DATA_DIR / kind / f"{kind.lower()}{d:%y%m%d}.txt"
    tried = st.session_state.setdefault("auto_fetched", set())
    # 当日の結果は場ごとに夜までかけて増えるので、30分たったら取り直す
    stale = kind == "K" and d == date.today() and path.exists() and time.time() - path.stat().st_mtime > 1800
    if (force or stale or not path.exists()) and (force or stale or (kind, d) not in tried) and d <= date.today():
        tried.add((kind, d))
        if (force or stale) and path.exists():
            path.unlink()
        try:
            with st.spinner("公式から取得しています…"):
                download_day(kind, d)
        except Exception as e:
            st.warning(f"公式から取得できませんでした: {e}")
    return path if path.exists() else None


# ---- 出走表 ----
if uploaded is not None:
    blob = uploaded.getvalue()
    text = extract_lzh(blob) if uploaded.name.endswith(".lzh") else blob.decode("cp932", errors="replace")
else:
    b_path = fetch_once("B", race_date)
    text = b_path.read_text(encoding="utf-8") if b_path else ""
entries = parse_b(text, race_date) if text else pd.DataFrame()

if entries.empty:
    st.markdown(ui.hero("競艇予想", f"{race_date.year}年{race_date.month}月{race_date.day}日"), unsafe_allow_html=True)
    st.info("この日の出走表はまだ公式に出ていません（たいてい前日の夕方以降に出ます）。")
    st.stop()

k_path = fetch_once("K", race_date, force=refresh_results)
results, payouts = parse_k(k_path.read_text(encoding="utf-8"), race_date) if k_path else (pd.DataFrame(), pd.DataFrame())

rule, gbm, gbm_ex = load_models()
venues = sorted(entries["venue"].unique())
c1, c2, c3 = st.columns([2, 2, 3])
venue = c1.selectbox("場", venues, format_func=lambda v: f"{VENUES.get(v, v)}")
races = sorted(entries.loc[entries["venue"] == venue, "race"].unique())
race = c2.selectbox("レース", races, format_func=lambda r: f"{r}R")
choices = (["LightGBM"] if gbm is not None else []) + ["ルール"]
model_col = c3.radio("予想に使うモデル", choices, horizontal=True,
                     help="LightGBM は過去のレースから学んだ予想、ルールは決まった計算式の予想です")

st.markdown(ui.hero(f"{VENUES.get(venue, venue)} {race}R",
                    f"{race_date.year}年{race_date.month}月{race_date.day}日・{model_col}で予想"), unsafe_allow_html=True)

# ---- 展示タイム: 結果ファイルがある日はそこから、まだのレースは直前情報ページから（このレースだけ） ----
ex_store = st.session_state.setdefault("exhibition", {})
odds_store = st.session_state.setdefault("odds", {})
key = (race_date, venue, race)
ex_cols = RACE_KEY + ["boat", "exhibition"]
from_results = results[ex_cols] if not results.empty else pd.DataFrame(columns=ex_cols)
race_done = ((from_results["venue"] == venue) & (from_results["race"] == race)).any()
is_live = not race_done and race_date == date.today()
if is_live:
    b1, b2, _ = st.columns([1, 1, 2])
    refetch = b1.button("展示タイムを取り直す", help="展示航走は締切のおよそ20〜30分前。終わると公式に出ます", width="stretch")
    get_odds = b2.button("オッズを取得", help="締切前のオッズを3ページ取るので少し時間がかかります", width="stretch")
    if refetch or key not in ex_store:
        try:
            ex_store[key] = beforeinfo.fetch(race_date, venue, race)
        except Exception as e:
            st.warning(f"直前情報を取得できませんでした: {e}")
    if get_odds:
        try:
            with st.spinner("オッズを取得しています…"):
                odds_store[key] = (odds.fetch(race_date, venue, race), time.strftime("%H:%M"))
        except Exception as e:
            st.warning(f"オッズを取得できませんでした: {e}")
from_before = [v.assign(date=k[0], venue=k[1], race=k[2]) for k, v in ex_store.items() if k[0] == race_date]
ex_all = pd.concat([from_results] + [x[ex_cols] for x in from_before], ignore_index=True).drop_duplicates(
    RACE_KEY + ["boat"], keep="first")

# ---- 予想 ----
df = add_features(entries)
local_k = tuple(sorted(str(p) for p in (DATA_DIR / "K").glob("k*.txt")))
df = add_history(df, load_history(local_k))
ex_all = ex_all.astype({"race": int, "boat": int, "exhibition": float})
df = add_exhibition(df.drop(columns="exhibition").merge(ex_all, on=RACE_KEY + ["boat"], how="left"))
# 6艇とも展示タイムがそろったレースだけ「展示あり」で予想する
has_ex = df.groupby(RACE_KEY)["ex_diff"].transform(lambda x: x.notna().all())
df["ルール"] = rule.predict(df).values
if gbm is not None:
    p = gbm.predict(df)
    if gbm_ex is not None and has_ex.any():
        p = p.where(~has_ex, gbm_ex.predict(df))
    df["LightGBM"] = p.values

g = df[(df["venue"] == venue) & (df["race"] == race)].sort_values("boat")
race_odds, odds_time = odds_store.get(key, ({}, None))
status = [ui.pill("展示タイム反映済み", "ok") if has_ex.loc[g.index].all() else ui.pill("展示タイム待ち", "wait")]
if race_odds:
    status.append(ui.pill(f"オッズ {odds_time} 時点", "info"))
elif is_live:
    status.append(ui.pill("オッズ未取得", "wait"))
if race_done:
    status.append(ui.pill("結果確定", "info"))
st.markdown(" ".join(status), unsafe_allow_html=True)

st.markdown(ui.section("各艇の1着確率"), unsafe_allow_html=True)
st.markdown(ui.boats_table(g, model_col), unsafe_allow_html=True)

race_pay = payouts[(payouts["venue"] == venue) & (payouts["race"] == race)] if not payouts.empty else payouts
hit_combos = {(r.ticket, r.combo): r.payout for r in race_pay.itertuples()}
all_probs = ticket_probs(dict(zip(g["boat"], g[model_col])))


def pick_info(t: str, c: str, p: float) -> dict:
    o = race_odds.get(t, {}).get(c)
    ev = p * o if o else None
    return {"combo": c, "prob": p, "odds": o, "ev": ev, "hit": hit_combos.get((t, c)),
            "value": ev is not None and ev >= ev_line and p >= min_prob}


st.markdown(ui.section("おすすめの買い目（当たりやすい順）"), unsafe_allow_html=True)
cols = st.columns(len(TICKET_TYPES))
for col, t in zip(cols, TICKET_TYPES):
    top = sorted(all_probs[t].items(), key=lambda kv: -kv[1])[:top_n]
    col.markdown(ui.ticket_card(t, [pick_info(t, c, p) for c, p in top]), unsafe_allow_html=True)

st.markdown(ui.section("妙味のある買い目（確率のわりに配当が高い順）"), unsafe_allow_html=True)
if race_odds:
    cands = [pick_info(t, c, p) | {"ticket": t} for t in TICKET_TYPES for c, p in all_probs[t].items()]
    vals = sorted([x for x in cands if x["value"]], key=lambda x: -x["ev"])[:8]
    if vals:
        vcols = st.columns(len(TICKET_TYPES))
        for col, t in zip(vcols, TICKET_TYPES):
            mine = [x for x in vals if x["ticket"] == t]
            if mine:
                col.markdown(ui.ticket_card(t, mine), unsafe_allow_html=True)
    else:
        st.caption(f"期待値 {ev_line:.2f} 以上（確率 {min_prob:.1%} 以上）の買い目はありません。見送りも選択肢です。")
    st.caption("期待値 = 予想の確率 × オッズ。1を超えるほど、確率のわりに配当が高い買い目です。オッズは締切まで動くので、直前に取り直してください。")
elif is_live:
    st.caption("「オッズを取得」を押すと、確率とオッズを比べて妙味のある買い目に印を付けます。")
else:
    st.caption("オッズは当日の締切前だけ取得できます。")

# ---- 結果 ----
st.markdown(ui.section("実際の結果"), unsafe_allow_html=True)
race_res = results[(results["venue"] == venue) & (results["race"] == race)] if not results.empty else results
if race_res.empty:
    st.caption("このレースの結果はまだ公式に出ていません。結果は場ごとに、全レース終了後に公開されます。")
else:
    order = race_res.sort_values("position", na_position="last")
    st.markdown(ui.result_places(order, dict(zip(g["boat"], g["name"])), dict(zip(g["boat"], g[model_col]))),
                unsafe_allow_html=True)
    pcols = st.columns(len(TICKET_TYPES))
    for col, t in zip(pcols, TICKET_TYPES):
        col.markdown(ui.payout_card(t, race_pay[race_pay["ticket"] == t]), unsafe_allow_html=True)

# この日の全レースで、いまの設定（モデルと点数）どおりに100円ずつ買った場合の成績
if not payouts.empty:
    done = df.merge(payouts[RACE_KEY].drop_duplicates(), on=RACE_KEY)
    if not done.empty:
        st.markdown(ui.section(f"この日の成績（{model_col}・券種ごとに上位{top_n}点を100円ずつ）"), unsafe_allow_html=True)
        summary = evaluate(done.reset_index(drop=True), done[model_col], payouts, top_n)
        kcols = st.columns(len(TICKET_TYPES))
        for col, t in zip(kcols, TICKET_TYPES):
            s = summary.loc[t]
            tone = "good" if s["回収率"] >= 1 else "bad"
            col.markdown(ui.kpi(f"{t}（{int(s['races'])}レース）", f"回収率 {s['回収率']:.0%}", tone)
                         + f'<div class="note" style="margin-top:6px">的中率 {s["的中率"]:.0%}・'
                           f'{int(s["cost"]):,}円 → {int(s["return"]):,}円</div>', unsafe_allow_html=True)

st.markdown('<div class="note" style="margin-top:24px">確率は出走表・展示タイム・過去の傾向から計算した目安です。'
            '舟券の購入は自己責任で、無理のない範囲で楽しんでください。</div>', unsafe_allow_html=True)
