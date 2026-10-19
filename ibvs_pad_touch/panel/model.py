from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from ibvs_pad_touch.config import load_panel_config


@dataclass(frozen=True)
class Rect:
    center_m: tuple[float, float]
    size_m: tuple[float, float]

    @property
    def cx(self) -> float:
        return float(self.center_m[0])

    @property
    def cy(self) -> float:
        return float(self.center_m[1])

    @property
    def width_m(self) -> float:
        return float(self.size_m[0])

    @property
    def height_m(self) -> float:
        return float(self.size_m[1])

    @property
    def left_m(self) -> float:
        return self.cx - 0.5 * self.width_m

    @property
    def right_m(self) -> float:
        return self.cx + 0.5 * self.width_m

    @property
    def top_m(self) -> float:
        return self.cy - 0.5 * self.height_m

    @property
    def bottom_m(self) -> float:
        return self.cy + 0.5 * self.height_m

    def corners_m(self) -> list[tuple[float, float]]:
        return [
            (self.left_m, self.top_m),
            (self.right_m, self.top_m),
            (self.right_m, self.bottom_m),
            (self.left_m, self.bottom_m),
        ]

    def to_xyz(self, z_m: float = 0.0) -> tuple[float, float, float]:
        return self.cx, self.cy, float(z_m)


@dataclass(frozen=True)
class Button:
    name: str
    button_id: int
    label: str
    rect: Rect
    color: str = "yellow"
    poke: bool = True
    aliases: tuple[str, ...] = ()
    source: str = "tape"

    def center_norm(self, panel: "PanelModel") -> tuple[float, float]:
        return panel.to_norm(self.rect.cx, self.rect.cy)

    def size_norm(self, panel: "PanelModel") -> tuple[float, float]:
        return self.rect.width_m / panel.width_m, self.rect.height_m / panel.height_m


@dataclass
class PanelModel:
    """Scale model of the fixed STULZ faceplate. Stage-1 poke set is Button 1/2/3."""

    width_m: float
    height_m: float
    buttons: dict[str, Button]
    display: Optional[Rect] = None
    references: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_norm(self, x_m: float, y_m: float) -> tuple[float, float]:
        return float(x_m) / self.width_m, float(y_m) / self.height_m

    def from_norm(self, u: float, v: float) -> tuple[float, float]:
        return float(u) * self.width_m, float(v) * self.height_m

    def poke_buttons(self) -> list[Button]:
        return [item for item in self.buttons.values() if item.poke]

    def button(self, key: int | str) -> Button:
        if isinstance(key, int):
            for item in self.buttons.values():
                if item.button_id == key:
                    return item
            raise KeyError(key)
        text = str(key).strip().lower().replace(" ", "_")
        if text in self.buttons:
            return self.buttons[text]
        if text.startswith("button_"):
            return self.button(int(text.split("_")[1]))
        if text.isdigit():
            return self.button(int(text))
        for item in self.buttons.values():
            names = {item.name.lower(), item.label.lower(), *[alias.lower() for alias in item.aliases]}
            if text in {name.replace(" ", "_") for name in names} or key.strip().lower() in names:
                return item
        raise KeyError(key)

    def poke_point_m(self, key: int | str) -> tuple[float, float, float]:
        return self.button(key).rect.to_xyz(0.0)

    def canonical_corners_m(self) -> list[tuple[float, float]]:
        return [
            (0.0, 0.0),
            (self.width_m, 0.0),
            (self.width_m, self.height_m),
            (0.0, self.height_m),
        ]

    def validate(self) -> None:
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("panel size must be positive")
        ids = [item.button_id for item in self.poke_buttons()]
        if sorted(ids) != [1, 2, 3]:
            raise ValueError(f"stage-1 poke set must be Button 1/2/3, got {ids}")
        left = self.button(1)
        right = self.button(2)
        ok = self.button(3)
        if left.rect.cx >= right.rect.cx:
            raise ValueError("Button 1 must be left of Button 2")
        if abs(left.rect.cy - right.rect.cy) > 1e-6:
            raise ValueError("Button 1/2 must share the same row")
        if ok.rect.cy <= left.rect.cy:
            raise ValueError("Button 3 must sit below Button 1/2")


def _rect_from_item(item: dict[str, Any]) -> Rect:
    return Rect(center_m=tuple(item["center_m"]), size_m=tuple(item["size_m"]))


def _buttons_from_raw(raw: dict[str, Any]) -> dict[str, Button]:
    buttons: dict[str, Button] = {}
    for name, item in (raw.get("buttons") or {}).items():
        buttons[name] = Button(
            name=name,
            button_id=int(item["id"]),
            label=str(item.get("label", name)),
            rect=_rect_from_item(item),
            color=str(item.get("color", "yellow")),
            poke=bool(item.get("poke", True)),
            aliases=tuple(item.get("aliases") or []),
            source=str(item.get("source", "tape")),
        )
    return buttons


def load_panel(path: str | Path | None = None) -> PanelModel:
    raw = load_panel_config(path)
    display_raw = raw.get("display") or {}
    display = None
    if display_raw:
        display = Rect(center_m=_display_center(display_raw), size_m=tuple(display_raw["size_m"]))
    model = PanelModel(
        width_m=float(raw["width_m"]),
        height_m=float(raw["height_m"]),
        buttons=_buttons_from_raw(raw),
        display=display,
        references=dict(raw.get("references") or {}),
        raw=raw,
    )
    model.validate()
    return model


def _display_center(display_raw: dict[str, Any]) -> tuple[float, float]:
    if "center_m" in display_raw:
        return tuple(display_raw["center_m"])
    ox, oy = display_raw["origin_m"]
    sx, sy = display_raw["size_m"]
    return ox + 0.5 * sx, oy + 0.5 * sy


def iter_named_rects(model: PanelModel) -> Iterable[tuple[str, Rect, dict[str, Any]]]:
    if model.display is not None:
        yield "display", model.display, {"label": "LCD", "color": "lcd"}
    for button in model.buttons.values():
        yield button.name, button.rect, {"label": button.label, "color": button.color, "poke": button.poke}
    for name, item in model.references.items():
        yield name, _rect_from_item(item), item
