"""LLM Hybrid connection probe and tip-delta parsing."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from ibvs_pad_touch.web.llm_fine import (
    CONNECT_TIMEOUT_S,
    _extract_json,
    align_delta,
    check_llm_connection,
    clip_delta,
    focus_press,
    image_still_off,
    keep_pressing,
    message_text,
    panel_scene,
    parse_tip_command,
)


class LlmFineTest(unittest.TestCase):
    def test_connect_timeout_default_is_sixty_seconds(self):
        self.assertEqual(CONNECT_TIMEOUT_S, 60.0)

    def test_check_connection_passes_timeout_to_urlopen(self):
        class _Resp:
            status = 200

            def read(self, _n=None):
                return b'{"data":[{"id":"Qwen3.8-27B-FP8"}]}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with patch("ibvs_pad_touch.web.llm_fine.urllib.request.urlopen", return_value=_Resp()) as opener:
            ok, detail = check_llm_connection("http://127.0.0.1:8000/v1", timeout_s=60.0)
        self.assertTrue(ok)
        self.assertIn("已连通", detail)
        self.assertEqual(opener.call_args.kwargs["timeout"], 60.0)

    def test_check_connection_reports_failure(self):
        import urllib.error

        with patch(
            "ibvs_pad_touch.web.llm_fine.urllib.request.urlopen",
            side_effect=urllib.error.URLError("timed out"),
        ):
            ok, detail = check_llm_connection("http://127.0.0.1:9/v1", timeout_s=60.0)
        self.assertFalse(ok)
        self.assertIn("连接失败", detail)

    def test_clip_and_parse_tip_command(self):
        clipped = clip_delta([0.05, -0.02, 0.001])
        self.assertEqual(clipped[0], 0.015)
        self.assertEqual(clipped[1], -0.015)
        parsed = parse_tip_command({"delta_xyz": [0.01, 0.0, -0.005], "done": False, "reason": "up"})
        self.assertEqual(parsed["delta_xyz"], [0.01, 0.0, -0.005])
        self.assertFalse(parsed["done"])

    def test_thinking_trace_still_yields_the_delta(self):
        text = message_text(
            {
                "content": None,
                "reasoning_content": '先看画面。<think>手指偏上</think>\n{"delta_xyz":[0,0,-0.004],"done":false,"reason":"down"}',
            }
        )
        parsed = parse_tip_command(_extract_json(text))
        self.assertEqual(parsed["delta_xyz"][2], -0.004)

    def test_upward_command_is_rejected_when_the_tip_is_already_above(self):
        # Button minus tip: Z is -20 mm, so the fingertip is already too high.
        aligned = align_delta([0.0, 0.008, 0.005], [0.01, 0.004, -0.020])
        self.assertLess(aligned[2], 0.0)
        self.assertGreaterEqual(aligned[2], -0.015)
        self.assertGreaterEqual(aligned[1], 0.0)

    def test_done_is_refused_while_x_is_still_short(self):
        step, done = keep_pressing([0.0, 0.0, 0.0], [0.025, 0.0, -0.01], True)
        self.assertFalse(done)
        self.assertGreater(step[0], 0.0)
        self.assertLessEqual(step[0], 0.015)
        close, finished = keep_pressing([0.0, 0.0, 0.0], [0.004, 0.0, 0.0], True)
        self.assertTrue(finished)
        self.assertEqual(close[0], 0.0)

    def test_finger_on_the_button_presses_forward_15_mm(self):
        step, done = focus_press(
            [0.0, 0.006, -0.005],
            [0.02, 0.02, 0.02],
            True,
            finger_on_button=True,
        )
        self.assertFalse(done)
        self.assertEqual(step, [0.015, 0.0, 0.0])

    def test_yellow_and_red_means_the_finger_is_too_low(self):
        scene = panel_scene(
            target="button_1",
            finger_uv=(200, 300),
            button_uv=None,
            yellow_bars=[{"cx": 160, "cy": 80, "left": 40, "right": 280}],
            red_bar=True,
        )
        self.assertTrue(scene["finger_low"])
        self.assertFalse(scene["finger_on_button"])
        self.assertIn("红条", scene["scene"])
        step, done = focus_press([0.01, 0.0, 0.0], [0.01, 0.0, 0.0], True, finger_low=True)
        self.assertFalse(done)
        self.assertGreater(step[2], 0.0)
        self.assertEqual(step[0], 0.0)

    def test_upper_yellow_bar_splits_into_button_1_and_2(self):
        left = panel_scene(
            target="button_1",
            finger_uv=(70, 80),
            button_uv=None,
            yellow_bars=[
                {"cx": 160, "cy": 80, "left": 40, "right": 280},
                {"cx": 160, "cy": 180, "left": 80, "right": 240},
            ],
        )
        right = panel_scene(
            target="button_2",
            finger_uv=(250, 80),
            button_uv=None,
            yellow_bars=[
                {"cx": 160, "cy": 80, "left": 40, "right": 280},
                {"cx": 160, "cy": 180, "left": 80, "right": 240},
            ],
        )
        lower = panel_scene(
            target="button_3",
            finger_uv=(160, 180),
            button_uv=None,
            yellow_bars=[
                {"cx": 160, "cy": 80, "left": 40, "right": 280},
                {"cx": 160, "cy": 180, "left": 80, "right": 240},
            ],
        )
        self.assertEqual(left["aim_uv"], [100.0, 80.0])
        self.assertEqual(right["aim_uv"], [220.0, 80.0])
        self.assertEqual(lower["aim_uv"], [160.0, 180.0])
        self.assertTrue(lower["finger_on_button"])
        self.assertIn("Button1", left["scene"])
        self.assertIn("Button2", right["scene"])

    def test_visible_pixel_gap_keeps_the_lateral_step(self):
        vision = {"pixel_gap_px": 90.0}
        self.assertTrue(image_still_off(vision))
        step, done = focus_press(
            [0.0, 0.01, -0.004],
            [0.001, 0.002, 0.001],
            True,
            image_open=True,
        )
        self.assertFalse(done)
        self.assertAlmostEqual(step[1], 0.01)
        self.assertAlmostEqual(step[2], -0.004)


if __name__ == "__main__":
    unittest.main()
