from ibvs_pad_touch.panel.model import Button, PanelModel, Rect, load_panel

__all__ = ["Button", "PanelModel", "Rect", "load_panel", "render_panel"]


def __getattr__(name: str):
    if name == "render_panel":
        from ibvs_pad_touch.panel.render import render_panel

        return render_panel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
