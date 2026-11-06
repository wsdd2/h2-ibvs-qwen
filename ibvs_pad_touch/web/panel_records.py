# -*- coding: utf-8 -*-
"""SQLite history of panel photos, task status, and OCR reports."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ibvs_pad_touch.web.panel_report import PAGE_NAMES, build_report, ocr_items

Recognize = Callable[[Any], list[dict[str, Any]] | None]


def _stamp(moment: float | None) -> str:
    if not moment:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(moment)))


class PanelRecords:
    def __init__(self, root: Path | str, recognize: Recognize | None = None) -> None:
        self.root = Path(root)
        self.images = self.root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self._recognize = ocr_items if recognize is None else recognize
        self._lock = threading.Lock()
        self._task_id: int | None = None
        self._conn = sqlite3.connect(self.root / "records.sqlite", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                started_at REAL NOT NULL,
                started_text TEXT NOT NULL,
                finished_at REAL,
                finished_text TEXT,
                state TEXT NOT NULL,
                mode TEXT,
                stage TEXT,
                message TEXT,
                steps_json TEXT,
                status_json TEXT
            );
            CREATE TABLE IF NOT EXISTS screens (
                id INTEGER PRIMARY KEY,
                task_id INTEGER,
                captured_at REAL NOT NULL,
                captured_text TEXT NOT NULL,
                label TEXT,
                page_kind TEXT NOT NULL,
                page_name TEXT NOT NULL,
                selection_text TEXT,
                ocr_text TEXT,
                report_text TEXT,
                report_json TEXT,
                image_file TEXT
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def open_task(self, status: dict[str, Any] | None) -> int:
        status = dict(status or {})
        now = time.time()
        with self._lock:
            if self._task_id:
                self._close_unlocked("interrupted", status)
            cursor = self._conn.execute(
                """
                INSERT INTO tasks
                    (started_at, started_text, state, mode, stage, message, steps_json, status_json)
                VALUES (?, ?, 'running', ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    _stamp(now),
                    str(status.get("sequence_mode") or ""),
                    str(status.get("sequence_stage") or status.get("control_stage") or ""),
                    str(status.get("message") or ""),
                    json.dumps(list(status.get("button_sequence") or []), ensure_ascii=False),
                    json.dumps(status, ensure_ascii=False, default=str),
                ),
            )
            self._conn.commit()
            self._task_id = int(cursor.lastrowid)
            return self._task_id

    def close_task(self, reason: str, status: dict[str, Any] | None = None) -> None:
        if reason not in {"done", "stopped", "interrupted"}:
            reason = "stopped"
        with self._lock:
            self._close_unlocked(reason, dict(status or {}))

    def _close_unlocked(self, reason: str, status: dict[str, Any]) -> None:
        task_id = self._task_id
        if not task_id:
            return
        self._task_id = None
        now = time.time()
        self._conn.execute(
            """
            UPDATE tasks
               SET finished_at=?, finished_text=?, state=?, stage=?, message=?, status_json=?
             WHERE id=?
            """,
            (
                now,
                _stamp(now),
                reason,
                str(status.get("sequence_stage") or status.get("control_stage") or ""),
                str(status.get("message") or ""),
                json.dumps(status, ensure_ascii=False, default=str),
                task_id,
            ),
        )
        self._conn.commit()

    def submit_screen(
        self,
        image: Any,
        label: str,
        block: dict[str, Any] | None,
        status: dict[str, Any] | None,
    ) -> None:
        shot = None
        if image is not None and getattr(image, "size", 0):
            shot = image.copy()
        block_copy = None if not block else dict(block)
        status_copy = dict(status or {})
        threading.Thread(
            target=self.store_screen,
            args=(shot, str(label or ""), block_copy, status_copy),
            name="panel-record",
            daemon=True,
        ).start()

    def store_screen(
        self,
        image: Any,
        label: str,
        block: dict[str, Any] | None,
        status: dict[str, Any] | None,
    ) -> int:
        try:
            return self._store_screen(image, label, block, status)
        except Exception as exc:
            print(f"[ibvs-records] 屏幕入库失败: {exc}", flush=True)
            return 0

    def _store_screen(
        self,
        image: Any,
        label: str,
        block: dict[str, Any] | None,
        status: dict[str, Any] | None,
    ) -> int:
        status = dict(status or {})
        selection = str((block or {}).get("text") or "")
        if image is None or getattr(image, "size", 0) == 0:
            built = {
                "page_kind": "screen",
                "page_name": "屏幕",
                "report_text": "没有读到屏幕",
                "report": {},
                "ocr_text": "",
            }
            jpeg = None
        else:
            try:
                recognized = self._recognize(image)
            except Exception as exc:
                recognized = None
                print(f"[ibvs-records] 读屏失败: {exc}", flush=True)
            if recognized is None:
                built = {
                    "page_kind": "screen",
                    "page_name": "屏幕",
                    "report_text": "画面已保存。RapidOCR 没有装上，这次没有读出文字。",
                    "report": {},
                    "ocr_text": "",
                }
            else:
                built = build_report(list(recognized), getattr(image, "shape", None))
                if not built["report_text"] and not built["ocr_text"]:
                    built["report_text"] = "画面已保存。这一屏没有读出文字。"
            jpeg = _jpeg(image)
        now = time.time()
        with self._lock:
            task_id = self._task_id
            cursor = self._conn.execute(
                """
                INSERT INTO screens
                    (task_id, captured_at, captured_text, label, page_kind, page_name,
                     selection_text, ocr_text, report_text, report_json, image_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '')
                """,
                (
                    task_id,
                    now,
                    _stamp(now),
                    str(label or ""),
                    built["page_kind"],
                    built["page_name"],
                    selection,
                    str(built.get("ocr_text") or "")[:4000],
                    str(built.get("report_text") or ""),
                    json.dumps(built.get("report") or {}, ensure_ascii=False),
                ),
            )
            row_id = int(cursor.lastrowid)
            image_file = ""
            if jpeg:
                image_file = f"{row_id}.jpg"
                (self.images / image_file).write_bytes(jpeg)
                self._conn.execute("UPDATE screens SET image_file=? WHERE id=?", (image_file, row_id))
            if task_id:
                self._conn.execute(
                    "UPDATE tasks SET stage=?, message=?, status_json=? WHERE id=? AND state='running'",
                    (
                        str(status.get("sequence_stage") or label or ""),
                        str(status.get("message") or ""),
                        json.dumps(status, ensure_ascii=False, default=str),
                        task_id,
                    ),
                )
            self._conn.commit()
        page_name = built["page_name"]
        print(f"[ibvs-records] #{row_id} {label} {page_name}", flush=True)
        return row_id

    def overview(self, live: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            current = self._task_row(self._task_id) if self._task_id else None
            tasks = [self._task_dict(row) for row in self._conn.execute("SELECT * FROM tasks ORDER BY id DESC LIMIT 12")]
            screens = [
                self._screen_dict(row)
                for row in self._conn.execute("SELECT * FROM screens ORDER BY id DESC LIMIT 60")
            ]
        climate = [row for row in screens if row["page_kind"] == "climate"][:30]
        events = [row for row in screens if row["page_kind"] == "events"][:30]
        return {
            "ok": True,
            "live": dict(live or {}),
            "task": current,
            "tasks": tasks,
            "climate": climate,
            "events": events,
            "screens": screens,
        }

    def image_bytes(self, row_id: int) -> bytes | None:
        path = self.images / f"{int(row_id)}.jpg"
        if not path.is_file():
            return None
        return path.read_bytes()

    def _task_row(self, task_id: int | None) -> dict[str, Any] | None:
        if not task_id:
            return None
        row = self._conn.execute("SELECT * FROM tasks WHERE id=?", (int(task_id),)).fetchone()
        if row is None:
            return None
        return self._task_dict(row)

    def _task_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        state = str(row["state"] or "")
        names = {"running": "进行中", "done": "已完成", "stopped": "已停止", "interrupted": "被新任务替换"}
        try:
            steps = json.loads(row["steps_json"] or "[]")
        except json.JSONDecodeError:
            steps = []
        return {
            "id": int(row["id"]),
            "state": state,
            "state_text": names.get(state, state),
            "mode": row["mode"] or "",
            "stage": row["stage"] or "",
            "message": row["message"] or "",
            "started_text": row["started_text"] or "",
            "finished_text": row["finished_text"] or "",
            "steps": steps,
        }

    def _screen_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            report = json.loads(row["report_json"] or "{}")
        except json.JSONDecodeError:
            report = {}
        kind = str(row["page_kind"] or "screen")
        image = "" if not row["image_file"] else f"/media/{int(row['id'])}.jpg"
        return {
            "id": int(row["id"]),
            "task_id": None if row["task_id"] is None else int(row["task_id"]),
            "captured_text": row["captured_text"] or "",
            "label": row["label"] or "",
            "page_kind": kind,
            "page_name": row["page_name"] or PAGE_NAMES.get(kind, "屏幕"),
            "selection_text": row["selection_text"] or "",
            "ocr_text": row["ocr_text"] or "",
            "report_text": row["report_text"] or "",
            "report": report,
            "image": image,
        }


def _jpeg(image: Any) -> bytes | None:
    try:
        import cv2
    except Exception:
        return None
    try:
        ok, blob = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    except Exception:
        return None
    if not ok:
        return None
    return blob.tobytes()
