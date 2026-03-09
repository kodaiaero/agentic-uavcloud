"""
ドローン測量ファイルの内容を解析するツール群。
Gemini が自律的に呼び出して、ファイルの品質・整合性を確認する。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

# scan_files で設定される作業ディレクトリ
_target_dir: str = ""


def set_target_dir(path: str) -> None:
    global _target_dir
    _target_dir = path


def _resolve_path(filename: str, subdir: str = "") -> Path | None:
    """ファイル名をフルパスに解決する。"""
    if not _target_dir:
        return None
    base = Path(_target_dir)
    if subdir:
        base = base / subdir
    path = base / filename
    return path if path.is_file() else None


# ---------------------------------------------------------------------------
# Tool: MRK ファイル解析
# ---------------------------------------------------------------------------

@tool
def check_mrk_file(filename: str) -> dict:
    """MRK（タイムスタンプ）ファイルの内容を解析し、エントリ数・時間範囲・精度統計・座標範囲を返す。
    filenameにはMRKファイル名（例: 20220607_0405_ASTimestamp.MRK）を指定する。"""

    path = _resolve_path(filename)
    if not path:
        return {"error": f"ファイルが見つかりません: {filename}"}

    entries = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        try:
            entry = {
                "index": int(parts[0]),
                "gps_time": float(parts[1]),
                "lat": float(parts[6].replace(",Lat", "")),
                "lon": float(parts[7].replace(",Lon", "")),
                "ellh": float(parts[8].replace(",Ellh", "")),
                "acc_n": int(parts[3].replace(",N", "")),
                "acc_e": int(parts[4].replace(",E", "")),
            }
            entries.append(entry)
        except (ValueError, IndexError):
            continue

    if not entries:
        return {"error": "MRKファイルのパースに失敗しました"}

    lats = [e["lat"] for e in entries]
    lons = [e["lon"] for e in entries]
    heights = [e["ellh"] for e in entries]
    acc_n = [abs(e["acc_n"]) for e in entries]
    acc_e = [abs(e["acc_e"]) for e in entries]

    first_time = entries[0]["gps_time"]
    last_time = entries[-1]["gps_time"]
    duration_sec = last_time - first_time

    return {
        "filename": filename,
        "total_entries": len(entries),
        "time_range": {
            "first_gps_tow_sec": round(first_time, 3),
            "last_gps_tow_sec": round(last_time, 3),
            "duration_sec": round(duration_sec, 1),
            "duration_min": round(duration_sec / 60, 1),
        },
        "coordinate_range": {
            "lat_min": round(min(lats), 6),
            "lat_max": round(max(lats), 6),
            "lon_min": round(min(lons), 6),
            "lon_max": round(max(lons), 6),
            "height_min_m": round(min(heights), 1),
            "height_max_m": round(max(heights), 1),
        },
        "accuracy_mm": {
            "north_mean": round(sum(acc_n) / len(acc_n)),
            "north_max": max(acc_n),
            "east_mean": round(sum(acc_e) / len(acc_e)),
            "east_max": max(acc_e),
        },
        "interval_sec": round(duration_sec / max(len(entries) - 1, 1), 2),
    }


# ---------------------------------------------------------------------------
# Tool: OBS (RINEX) ファイル解析
# ---------------------------------------------------------------------------

@tool
def check_obs_file(filename: str, location: str = "root") -> dict:
    """OBS（RINEX観測データ）ファイルのヘッダーを解析し、観測時間・衛星システム・ファイルサイズを返す。
    filenameにはOBSファイル名を指定する。locationは 'root'（ドローンOBS）または 'base_station_logs'（基準局OBS）。"""

    subdir = "" if location == "root" else location
    path = _resolve_path(filename, subdir)
    if not path:
        return {"error": f"ファイルが見つかりません: {filename} (location={location})"}

    file_size_mb = round(path.stat().st_size / (1024 * 1024), 2)

    header_text = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            header_text += line
            if "END OF HEADER" in line:
                break

    # RINEX バージョン
    version = ""
    version_match = re.search(r"^\s*([\d.]+)\s+.*RINEX VERSION", header_text, re.MULTILINE)
    if version_match:
        version = version_match.group(1).strip()

    # 観測時間の抽出
    first_obs = _parse_rinex_time(header_text, "TIME OF FIRST OBS")
    last_obs = _parse_rinex_time(header_text, "TIME OF LAST OBS")

    duration_info = {}
    if first_obs and last_obs:
        delta = last_obs - first_obs
        duration_info = {
            "first_obs": first_obs.strftime("%Y-%m-%d %H:%M:%S"),
            "last_obs": last_obs.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": round(delta.total_seconds()),
            "duration_min": round(delta.total_seconds() / 60, 1),
        }

    # 衛星システム
    sat_systems = []
    for code, name in [("G", "GPS"), ("R", "GLONASS"), ("E", "Galileo"), ("C", "BeiDou"), ("J", "QZSS")]:
        if re.search(rf"^{code}\s+\d+", header_text, re.MULTILINE):
            sat_systems.append(name)

    # 基準局の近似位置
    approx_pos = {}
    pos_match = re.search(
        r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+APPROX POSITION XYZ",
        header_text,
        re.MULTILINE,
    )
    if pos_match:
        x, y, z = float(pos_match.group(1)), float(pos_match.group(2)), float(pos_match.group(3))
        if abs(x) > 1 or abs(y) > 1 or abs(z) > 1:
            approx_pos = {"x": x, "y": y, "z": z}

    result = {
        "filename": filename,
        "location": location,
        "file_size_mb": file_size_mb,
        "rinex_version": version,
        "satellite_systems": sat_systems,
    }
    if duration_info:
        result["observation_time"] = duration_info
    if approx_pos:
        result["approx_position_ecef"] = approx_pos

    return result


def _parse_rinex_time(header: str, label: str) -> datetime | None:
    """RINEXヘッダーから時刻を抽出する。"""
    match = re.search(
        rf"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+.*{label}",
        header,
        re.MULTILINE,
    )
    if not match:
        return None
    try:
        return datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
            int(float(match.group(6))),
        )
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Tool: データ整合性チェック
# ---------------------------------------------------------------------------

@tool
def validate_data_consistency() -> dict:
    """画像枚数とMRKタイムスタンプ数の整合性、OBS観測時間の十分性を自動チェックする。引数は不要。"""

    if not _target_dir:
        return {"error": "対象ディレクトリが設定されていません"}

    base = Path(_target_dir)
    issues: list[str] = []
    checks: list[str] = []

    # 画像数カウント
    photos_dir = base / "photos"
    image_count = 0
    if photos_dir.is_dir():
        image_count = len([f for f in photos_dir.iterdir() if f.is_file() and not f.name.startswith(".")])

    # MRK行数カウント
    mrk_files = [f for f in base.iterdir() if f.is_file() and f.suffix.upper() == ".MRK"]
    mrk_count = 0
    if mrk_files:
        for line in mrk_files[0].read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" in line:
                mrk_count += 1

    # 画像数 vs MRK数 チェック
    if image_count > 0 and mrk_count > 0:
        if image_count == mrk_count:
            checks.append(f"✓ 画像枚数({image_count})とMRKエントリ数({mrk_count})が一致")
        else:
            diff = abs(image_count - mrk_count)
            issues.append(f"⚠ 画像枚数({image_count})とMRKエントリ数({mrk_count})が不一致（差: {diff}）")
    elif image_count > 0 and mrk_count == 0:
        issues.append("⚠ 画像はあるがMRKファイルがありません（PPK処理不可）")

    # ドローンOBSの観測時間チェック
    drone_obs_files = [f for f in base.iterdir() if f.is_file() and f.suffix.lower() == ".obs"]
    for obs_file in drone_obs_files:
        header = ""
        with open(obs_file, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                header += line
                if "END OF HEADER" in line:
                    break
        first = _parse_rinex_time(header, "TIME OF FIRST OBS")
        last = _parse_rinex_time(header, "TIME OF LAST OBS")
        if first and last:
            dur_min = (last - first).total_seconds() / 60
            if dur_min < 1:
                issues.append(f"⚠ ドローンOBS ({obs_file.name}) の観測時間が{dur_min:.1f}分と極端に短い")
            else:
                checks.append(f"✓ ドローンOBS ({obs_file.name}) 観測時間: {dur_min:.1f}分")

    # 基準局OBSの観測時間チェック
    base_dir = base / "base_station_logs"
    if base_dir.is_dir():
        for obs_file in base_dir.iterdir():
            if obs_file.suffix.lower() != ".obs":
                continue
            header = ""
            with open(obs_file, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    header += line
                    if "END OF HEADER" in line:
                        break
            first = _parse_rinex_time(header, "TIME OF FIRST OBS")
            last = _parse_rinex_time(header, "TIME OF LAST OBS")
            if first and last:
                dur_min = (last - first).total_seconds() / 60
                if dur_min < 30:
                    issues.append(f"⚠ 基準局OBS ({obs_file.name}) の観測時間が{dur_min:.1f}分（推奨: 30分以上）")
                else:
                    checks.append(f"✓ 基準局OBS ({obs_file.name}) 観測時間: {dur_min:.1f}分")

    # OBSファイルのサイズチェック
    all_obs = list(drone_obs_files)
    if base_dir.is_dir():
        all_obs += [f for f in base_dir.iterdir() if f.suffix.lower() == ".obs"]
    for obs_file in all_obs:
        size_kb = obs_file.stat().st_size / 1024
        if size_kb < 10:
            issues.append(f"⚠ {obs_file.name} のファイルサイズが{size_kb:.1f}KBと極端に小さい（破損の可能性）")

    return {
        "passed_checks": checks,
        "issues": issues if issues else ["問題は検出されませんでした"],
        "summary": "問題なし" if not issues else f"{len(issues)}件の問題を検出",
    }
