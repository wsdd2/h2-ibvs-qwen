"""Type a conversation with the local Qwen server. Empty line exits."""

from __future__ import annotations

import json
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen3.8-27B-FP8"


def main() -> None:
    messages: list[dict[str, str]] = []
    print("和 Qwen 对话。空行回车退出。", flush=True)
    while True:
        try:
            text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            return
        messages.append({"role": "user", "content": text})
        body = json.dumps(
            {
                "model": MODEL,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": messages,
            }
        ).encode()
        request = urllib.request.Request(
            URL, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            reply = json.loads(response.read().decode())
        content = reply["choices"][0]["message"].get("content") or ""
        messages.append({"role": "assistant", "content": content})
        print("Qwen>", content, flush=True)


if __name__ == "__main__":
    main()
