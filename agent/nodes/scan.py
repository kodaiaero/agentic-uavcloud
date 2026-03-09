from __future__ import annotations

import sys
from pathlib import Path

from agent.state import FileInventory, RouteCapability, GraphState


def scan_files(state: GraphState) -> dict:
    """ディレクトリを走査し、ファイルをカテゴリ別に分類する。"""
    target = Path(state.target_dir)
    if not target.is_dir():
        print(f"\n[ERROR] ディレクトリが見つかりません: {target}")
        sys.exit(1)

    inv = FileInventory()

    photos_dir = target / "photos"
    if photos_dir.is_dir():
        inv.images = sorted(
            f.name for f in photos_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        )

    inv.drone_obs = sorted(
        f.name for f in target.iterdir()
        if f.is_file() and f.suffix.lower() == ".obs"
    )

    inv.timestamp = sorted(
        f.name for f in target.iterdir()
        if f.is_file() and f.suffix.upper() == ".MRK"
    )

    base_dir = target / "base_station_logs"
    if base_dir.is_dir():
        inv.base_obs = sorted(
            f.name for f in base_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".obs"
        )

    marker_dir = target / "aerobo_marker_logs"
    if marker_dir.is_dir():
        inv.markers = sorted(
            f.name for f in marker_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".log"
        )

    print("\n========== ファイルスキャン結果 ==========")
    print(f"  Images       : {len(inv.images)} 枚")
    print(f"  Drone OBS    : {len(inv.drone_obs)} 件")
    print(f"  Timestamp    : {len(inv.timestamp)} 件")
    print(f"  Base OBS     : {len(inv.base_obs)} 件")
    print(f"  Markers      : {len(inv.markers)} 件")
    print("==========================================")

    return {"inventory": inv}


def analyze_capability(state: GraphState) -> dict:
    """ファイル構成から各ルートの実行可否を判定する。"""
    inv = state.inventory

    cap = RouteCapability(
        route_a=len(inv.images) > 0,
        route_b=len(inv.drone_obs) > 0 and len(inv.base_obs) > 0,
        route_c=len(inv.images) > 0 and len(inv.drone_obs) > 0 and len(inv.base_obs) > 0 and len(inv.timestamp) > 0,
        route_d=len(inv.images) > 0 and len(inv.drone_obs) > 0 and len(inv.base_obs) > 0 and len(inv.timestamp) > 0 and len(inv.markers) > 0,
    )

    print("\n========== ルート判定結果 ==========")
    labels = {
        "route_a": "A: 簡易写真処理",
        "route_b": "B: 高精度基線解析",
        "route_c": "C: フルPPK写真処理",
        "route_d": "D: 精度検証付き処理",
    }
    for key, label in labels.items():
        status = "✓ 実行可能" if getattr(cap, key) else "✗ 条件未達"
        print(f"  [{status}] {label}")
    print("====================================")

    return {"capability": cap}
