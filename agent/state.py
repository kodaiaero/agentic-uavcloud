from __future__ import annotations

from pydantic import BaseModel, Field


class FileInventory(BaseModel):
    images: list[str] = Field(default_factory=list)
    drone_obs: list[str] = Field(default_factory=list)
    timestamp: list[str] = Field(default_factory=list)
    base_obs: list[str] = Field(default_factory=list)
    markers: list[str] = Field(default_factory=list)


class RouteCapability(BaseModel):
    route_a: bool = False  # 簡易写真処理
    route_b: bool = False  # 高精度基線解析
    route_c: bool = False  # フルPPK写真処理
    route_d: bool = False  # 精度検証付き処理


ROUTE_LABELS: dict[str, str] = {
    "A": "簡易写真処理",
    "B": "高精度基線解析",
    "C": "フルPPK写真処理",
    "D": "精度検証付き処理",
}


class GraphState(BaseModel):
    target_dir: str = ""
    inventory: FileInventory = Field(default_factory=FileInventory)
    capability: RouteCapability = Field(default_factory=RouteCapability)
    recommendation: str = ""
    # チャットループ用
    chat_history: list = Field(default_factory=list)
    user_input: str = ""
    next_action: str = ""  # "chat" | "rescan" | "execute" | "quit"
    selected_route: str = ""
    execution_result: str = ""
