# -*- coding: utf-8 -*-
"""Foreground start keeps one log per run and drops logs past the newest 50."""
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "session_log.sh"
START = ROOT / "scripts" / "start_web_remote.sh"


def _bash():
    return shutil.which("bash")


def _bash_path(path: Path, bash: str) -> str:
    text = path.as_posix()
    if "WindowsApps" in bash.replace("\\", "/") and len(path.drive) == 2:
        drive = path.drive[0].lower()
        rest = text.split(":", 1)[1]
        return f"/mnt/{drive}{rest}"
    return text


def test_start_script_saves_a_session_log_and_prints_it_on_exit():
    text = START.read_text(encoding="utf-8")
    assert "session_log_begin" in text
    assert "session_log_finish" in text
    assert 'trap session_log_finish EXIT' in text
    helper = HELPER.read_text(encoding="utf-8")
    assert "ibvs_${start}_${end}.log" in helper
    assert "ibvs_${start}_running.log" in helper
    assert "本次日志:" in helper


def test_prune_keeps_the_newest_forty_nine_session_logs(tmp_path):
    bash = _bash()
    if bash is None:
        return
    for index in range(53):
        path = tmp_path / f"ibvs_20260101_{index:06d}_running.log"
        path.write_text("x", encoding="utf-8")
        stamp = 1_700_000_000 + index
        os.utime(path, (stamp, stamp))
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
    helper = _bash_path(HELPER, bash)
    folder = _bash_path(tmp_path, bash)
    script = f'source "{helper}" && session_log_prune "{folder}" 49'
    completed = subprocess.run(
        [bash, "-c", script],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stderr
    left = sorted(path.name for path in tmp_path.glob("ibvs_*.log"))
    assert len(left) == 49
    assert "ibvs_20260101_000000_running.log" not in left
    assert "ibvs_20260101_000003_running.log" not in left
    assert "ibvs_20260101_000004_running.log" in left
    assert "ibvs_20260101_000052_running.log" in left
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert "已删除 4 条" in completed.stdout


def test_finish_renames_the_running_log_with_the_end_stamp(tmp_path):
    bash = _bash()
    if bash is None:
        return
    running = tmp_path / "ibvs_20261022_150000_running.log"
    running.write_text("hello\n", encoding="utf-8")
    helper = _bash_path(HELPER, bash)
    folder = _bash_path(tmp_path, bash)
    source = _bash_path(running, bash)
    script = "\n".join(
        [
            f'source "{helper}"',
            f'export IBVS_SESSION_LOG_DIR="{folder}"',
            'export IBVS_SESSION_START="20261022_150000"',
            f'export IBVS_SESSION_LOG="{source}"',
            'export IBVS_SESSION_LOG_FINISHED=0',
            "session_log_finish",
        ]
    )
    completed = subprocess.run(
        [bash, "-c", script],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stderr
    assert not running.exists()
    saved = list(tmp_path.glob("ibvs_20261022_150000_*.log"))
    assert len(saved) == 1
    assert "_running" not in saved[0].name
    assert saved[0].read_text(encoding="utf-8") == "hello\n"
    assert "本次日志:" in completed.stdout
    assert saved[0].name in completed.stdout


def test_begin_then_finish_records_both_stamps_in_the_file(tmp_path):
    bash = _bash()
    if bash is None:
        return
    helper = _bash_path(HELPER, bash)
    folder = _bash_path(tmp_path, bash)
    script = "\n".join(
        [
            f'source "{helper}"',
            f'session_log_begin "{folder}"',
            "echo hello-from-task",
            "session_log_finish",
        ]
    )
    completed = subprocess.run(
        [bash, "-c", script],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    assert completed.returncode == 0, completed.stderr
    assert not list(tmp_path.glob("*_running.log"))
    saved = list(tmp_path.glob("ibvs_*.log"))
    assert len(saved) == 1
    assert re.fullmatch(r"ibvs_\d{8}_\d{6}_\d{8}_\d{6}\.log", saved[0].name)
    text = saved[0].read_text(encoding="utf-8")
    assert "hello-from-task" in text
    assert f"本次日志: {folder}/{saved[0].name}" in text
    assert "本次日志:" in completed.stdout
    assert saved[0].name in completed.stdout
