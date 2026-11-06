# -*- coding: utf-8 -*-
"""Climate and event reports, and the customer record page."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ibvs_pad_touch.web.panel_records import PanelRecords
from ibvs_pad_touch.web.panel_report import build_report, ocr_items
from ibvs_pad_touch.web.records_site import build_records_app


def _climate_photo():
    return [
        {"text": "返回", "xywh": [14, 24, 122, 61]},
        {"text": "信息", "xywh": [241, 34, 133, 48]},
        {"text": "操作", "xywh": [471, 19, 135, 68]},
        {"text": "配置", "xywh": [700, 18, 134, 59]},
        {"text": "室内", "xywh": [585, 146, 123, 63]},
        {"text": "23.1C*", "xywh": [615, 190, 293, 100]},
        {"text": "57.0%", "xywh": [621, 264, 188, 71]},
        {"text": "Unit", "xywh": [171, 469, 98, 31]},
        {"text": "1 15:33", "xywh": [757, 468, 188, 46]},
    ], (536, 959, 3)


def _event_photo():
    return [
        {"text": "PM", "xywh": [613, 84, 97, 38]},
        {"text": "min", "xywh": [837, 91, 78, 26]},
        {"text": "设备超动", "xywh": [53, 123, 262, 72]},
        {"text": "15.09", "xywh": [782, 129, 131, 30]},
        {"text": "1217", "xywh": [621, 130, 131, 32]},
        {"text": "非本地停机", "xywh": [53, 184, 327, 69]},
        {"text": "1519", "xywh": [781, 189, 131, 31]},
        {"text": "1217", "xywh": [623, 191, 130, 30]},
        {"text": "开机", "xywh": [56, 245, 129, 70]},
        {"text": "1217", "xywh": [624, 252, 128, 30]},
        {"text": "15109", "xywh": [781, 252, 131, 30]},
        {"text": "关机", "xywh": [61, 306, 126, 68]},
        {"text": "1217", "xywh": [624, 312, 128, 31]},
        {"text": "15:98", "xywh": [781, 312, 132, 30]},
        {"text": "设备停止", "xywh": [55, 361, 268, 78]},
        {"text": "1217", "xywh": [624, 371, 129, 32]},
        {"text": "15106", "xywh": [782, 371, 128, 31]},
        {"text": "115", "xywh": [762, 464, 107, 43]},
        {"text": "unit", "xywh": [173, 472, 97, 30]},
        {"text": "返回", "xywh": [18, 27, 130, 61]},
    ], (534, 960, 3)


def test_status_page_report_keeps_indoor_temperature_and_humidity():
    items, shape = _climate_photo()
    report = build_report(items, shape)
    assert report["page_kind"] == "climate"
    assert report["page_name"] == "温湿度现状"
    assert report["report"]["temperature_c"] == 23.1
    assert report["report"]["humidity_pct"] == 57.0
    assert "室内温度 23.1 ℃" in report["report_text"]
    assert "室内湿度 57.0 %" in report["report_text"]
    assert "15:33" not in report["report_text"]


def test_home_screen_is_not_a_climate_report():
    items = [
        {"text": "22°C", "xywh": [143, 24, 126, 49]},
        {"text": "ON", "xywh": [277, 24, 86, 98]},
        {"text": "57%", "xywh": [143, 73, 126, 49]},
        {"text": "ME", "xywh": [32, 54, 54, 40]},
    ]
    report = build_report(items, (538, 960, 3))
    assert report["page_kind"] == "screen"
    assert report["report"] == {}


def test_menu_item_named_event_log_is_not_the_event_page():
    items = [
        {"text": "数据记录", "xywh": [40, 80, 180, 40]},
        {"text": "事件记录", "xywh": [40, 140, 180, 40]},
        {"text": "运行时间", "xywh": [40, 200, 180, 40]},
    ]
    report = build_report(items, (400, 640, 3))
    assert report["page_kind"] == "screen"


def test_event_page_pairs_each_name_with_the_date_on_its_row():
    items, shape = _event_photo()
    report = build_report(items, shape)
    assert report["page_kind"] == "events"
    rows = report["report"]["rows"]
    assert [row["name"] for row in rows] == ["设备起动", "非本地停机", "开机", "关机", "设备停止"]
    assert rows[0]["when"] == "12/17 15:09"
    assert rows[1]["when"] == "12/17 15:19"
    assert rows[2]["when"] == "12/17 15:10"
    assert rows[3]["when"] == "12/17 15:98"
    assert rows[4]["when"] == "12/17 15:10"
    assert report["report_text"].splitlines()[1] == "1. 设备起动  12/17 15:09"


def test_records_page_serves_the_report_and_the_photo():
    items = [
        {"text": "室内", "xywh": [4, 2, 12, 6]},
        {"text": "23.1C", "xywh": [4, 10, 18, 6]},
        {"text": "57.0%", "xywh": [4, 18, 16, 6]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        records = PanelRecords(Path(tmp), recognize=lambda _image: items)
        try:
            task_id = records.open_task(
                {"message": "组合启动", "sequence_mode": "direct_pbvs", "button_sequence": ["button_1"]}
            )
            image = np.zeros((32, 48, 3), dtype=np.uint8)
            row_id = records.store_screen(image, "Button1", {"text": "信息", "score": 1.0}, {"message": "Button1"})
            assert row_id > 0
            blank = records.store_screen(None, "Button2", None, {"message": "没有读到屏幕"})
            assert blank > 0
            records.close_task("done", {"message": "动作组合完成"})
            second = records.open_task({"message": "再次组合"})
            third = records.open_task({"message": "替换"})
            assert third != task_id and third != second
            overview = records.overview({"sequence_run": False, "message": "空闲", "running": False})
            assert overview["climate"][0]["report"]["temperature_c"] == 23.1
            assert overview["climate"][0]["selection_text"] == "信息"
            assert overview["screens"][0]["report_text"] == "没有读到屏幕"
            assert any(task["state"] == "interrupted" for task in overview["tasks"])
            assert any(task["state"] == "done" for task in overview["tasks"])
            app = build_records_app(records, lambda: {"message": "空闲", "sequence_run": False, "running": False})
            client = app.test_client()
            page = client.get("/")
            assert page.status_code == 200
            assert "温湿度现状" in page.get_data(as_text=True)
            body = client.get("/api/overview").get_json()
            assert body["climate"][0]["page_name"] == "温湿度现状"
            photo = client.get(body["climate"][0]["image"])
            assert photo.status_code == 200
            assert photo.headers["Content-Type"].startswith("image/jpeg")
        finally:
            records.close()


def test_missing_rapidocr_still_saves_the_screen():
    with tempfile.TemporaryDirectory() as tmp:
        records = PanelRecords(Path(tmp), recognize=lambda _image: None)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        try:
            records.store_screen(image, "动作前", {"text": "ON"}, {"message": "动作前"})
            saved = records.overview()["screens"][0]
            assert saved["page_kind"] == "screen"
            assert "RapidOCR 没有装上" in saved["report_text"]
            assert saved["image"].endswith(".jpg")
        finally:
            records.close()


def test_screen_read_is_handed_to_the_record_callback():
    from ibvs_pad_touch.web.arm_motion import ArmMover

    seen = []
    mover = ArmMover(telemetry=lambda: {}, confirm=True, log=lambda _message: None)
    mover.screen_settle_s = 0
    mover.read_screen = lambda: np.full((40, 40, 3), 180, dtype=np.uint8)
    mover.on_screen_record = lambda image, label, info: seen.append((label, image is not None))
    mover._read_progress_screen("动作前")
    mover.read_screen = lambda: None
    mover._read_progress_screen("Button1")
    assert seen == [("动作前", True), ("Button1", False)]
    begin = []
    mover.on_task_begin = lambda: begin.append("begin")
    mover.on_task_end = lambda reason: begin.append(reason)
    mover._emit_task_begin()
    mover._emit_task_end("done")
    assert begin == ["begin", "done"]


def test_operator_page_points_at_the_records_port():
    from ibvs_pad_touch.web.dashboard import PAGE_HTML

    html = PAGE_HTML.replace("__RECORDS_PORT__", "8082")
    assert 'id="recordsLink"' in html
    assert "const recordsPort = 8082;" in html


def test_ocr_parser_reads_a_rapidocr_tuple():
    from ibvs_pad_touch.web.panel_report import _items_from_outcome

    outcome = (
        [
            [[[10, 20], [40, 20], [40, 50], [10, 50]], "室内", 0.8],
        ],
        None,
    )
    items = _items_from_outcome(outcome)
    assert items[0]["text"] == "室内"
    assert items[0]["xywh"] == [10.0, 20.0, 30.0, 30.0]
    assert ocr_items(None) == []


def main() -> None:
    tests = [
        test_status_page_report_keeps_indoor_temperature_and_humidity,
        test_home_screen_is_not_a_climate_report,
        test_menu_item_named_event_log_is_not_the_event_page,
        test_event_page_pairs_each_name_with_the_date_on_its_row,
        test_records_page_serves_the_report_and_the_photo,
        test_missing_rapidocr_still_saves_the_screen,
        test_screen_read_is_handed_to_the_record_callback,
        test_operator_page_points_at_the_records_port,
        test_ocr_parser_reads_a_rapidocr_tuple,
    ]
    for test in tests:
        test()
        print("ok", test.__name__)


if __name__ == "__main__":
    main()
