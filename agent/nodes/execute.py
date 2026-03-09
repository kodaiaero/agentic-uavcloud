from __future__ import annotations

from agent.state import GraphState


def execute_mock(state: GraphState) -> dict:
    """選択されたルートをモック実行する。"""
    route = state.selected_route
    inv = state.inventory

    print(f"\n========== ルート{route} モック実行 ==========")

    if route == "A":
        print(f"[1/3] {len(inv.images)} 枚の画像を読み込み中...")
        print("[2/3] SfM (Structure from Motion) 処理を実行中...")
        print("[3/3] オルソ画像・点群データを生成中...")
        result = f"簡易写真処理が完了しました。{len(inv.images)}枚の画像からオルソ画像を生成。"

    elif route == "B":
        print(f"[1/3] ドローンOBS ({inv.drone_obs[0]}) を読み込み中...")
        print(f"[2/3] 基準局OBS ({inv.base_obs[0]}) を読み込み中...")
        print("[3/3] RTKLib による基線解析を実行中...")
        result = "高精度基線解析が完了しました。PPK座標を算出。"

    elif route == "C":
        print(f"[1/5] {len(inv.images)} 枚の画像を読み込み中...")
        print(f"[2/5] ドローンOBS ({inv.drone_obs[0]}) を読み込み中...")
        print(f"[3/5] 基準局OBS ({inv.base_obs[0]}) を読み込み中...")
        print(f"[4/5] タイムスタンプ ({inv.timestamp[0]}) でPPK測位を実行中...")
        print("[5/5] PPK座標付きSfM処理を実行中...")
        result = f"フルPPK写真処理が完了しました。{len(inv.images)}枚をPPK補正済み座標で処理。"

    elif route == "D":
        print(f"[1/6] {len(inv.images)} 枚の画像を読み込み中...")
        print(f"[2/6] ドローンOBS ({inv.drone_obs[0]}) を読み込み中...")
        print(f"[3/6] 基準局OBS ({inv.base_obs[0]}) を読み込み中...")
        print(f"[4/6] タイムスタンプ ({inv.timestamp[0]}) でPPK測位を実行中...")
        print("[5/6] PPK座標付きSfM処理を実行中...")
        print(f"[6/6] {len(inv.markers)} 個のマーカーで精度検証を実行中...")
        result = (
            f"精度検証付き処理が完了しました。"
            f"{len(inv.images)}枚をPPK補正済みで処理し、"
            f"{len(inv.markers)}点のマーカーで精度検証。"
        )
    else:
        result = "不明なルートです。"

    print(f"\n[結果] {result}")
    print("=" * 44)

    return {"execution_result": result}
