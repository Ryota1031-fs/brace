# 競艇予想（試作）

公式の番組表・競走成績ファイルから各艇の1着確率を出し、2連単・2連複・3連単・3連複の買い目確率を表示する自分用のWebアプリ。

## 使い方
```bash
pip install -r requirements.txt

# 1. 過去データを取得（公式サイトの負荷を避けるため1ファイルごとに2秒待つ）
python -m boatrace_yosou.download 2025-01-01 2026-10-16

# 2. ルールベースとLightGBMを学習し、検証期間の的中率・回収率を比べる（models/ にモデルを保存）
python -m boatrace_yosou.backtest --top 5

# 3. 画面を起動
streamlit run app.py
```

## 構成
- `boatrace_yosou/download.py` 公式ファイル（LZH）の取得と展開
- `boatrace_yosou/parse.py` 番組表(B)・競走成績(K)を表に変換
- `boatrace_yosou/model.py` 特徴量、ルールベース、LightGBM、Harville法による買い目確率
- `boatrace_yosou/backtest.py` 日付で学習と検証に分けた的中率・回収率の集計
- `app.py` Streamlitの画面

## 注意
- 公式サイトのサイトポリシーに従い、私的使用の範囲で使う。取得データの再配布はしない。
- 解析処理は公式レイアウトをまねた見本でしか確認できていない。実ファイルで読めない行があれば `parse.py` の正規表現を直す。
