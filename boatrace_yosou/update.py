"""GitHub に置いた最新版をダウンロードして、このフォルダを上書きする。

data/（取り込んだ出走表・結果）は触らない。git は不要で、公開リポジトリの zip を取りにいく。
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

REPO_ZIP = "https://github.com/Ryota1031-fs/brace/archive/refs/heads/main.zip"
ROOT = Path(__file__).resolve().parent.parent
KEEP = {"data", ".git", ".venv"}


def update(root: Path = ROOT) -> list[str]:
    """更新したファイルの相対パスを返す。"""
    resp = requests.get(REPO_ZIP, timeout=60)
    resp.raise_for_status()
    changed = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for info in z.infolist():
            # zip の中は "brace-main/..." の形なので先頭のフォルダを外す
            rel = Path(*Path(info.filename).parts[1:])
            if info.is_dir() or not rel.parts or rel.parts[0] in KEEP:
                continue
            target = root / rel
            new = z.read(info)
            if target.exists() and target.read_bytes() == new:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(new)
            changed.append(rel.as_posix())
    return changed


if __name__ == "__main__":
    files = update()
    print("更新しました:\n" + "\n".join(files) if files else "すでに最新です")
