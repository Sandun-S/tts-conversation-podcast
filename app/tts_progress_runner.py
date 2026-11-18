"""Run podcastfy.client with per-turn [TTS] progress logging."""

import re
import runpy
import sys
from pathlib import Path


def _estimate_transcript_turns() -> int | None:
    if "--transcript" not in sys.argv:
        return None
    idx = sys.argv.index("--transcript")
    if idx + 1 >= len(sys.argv):
        return None
    path = Path(sys.argv[idx + 1])
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    tags = re.findall(r"<Person[12]>", text)
    if tags:
        return len(tags)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) or None


def _install_tts_hooks(total: int | None) -> None:
    state = {"count": 0}

    def _log_turn() -> None:
        state["count"] += 1
        if total:
            print(f"[TTS] {state['count']}/{total}", flush=True)
        else:
            print(f"[TTS] {state['count']}", flush=True)

    try:
        from openai.resources.audio.speech import Speech

        orig_create = Speech.create

        def wrapped_create(self, *args, **kwargs):
            _log_turn()
            return orig_create(self, *args, **kwargs)

        Speech.create = wrapped_create
    except Exception:
        pass

    try:
        import edge_tts

        orig_save = edge_tts.Communicate.save

        async def wrapped_save(self, *args, **kwargs):
            _log_turn()
            return await orig_save(self, *args, **kwargs)

        edge_tts.Communicate.save = wrapped_save
    except Exception:
        pass


def main() -> None:
    total = _estimate_transcript_turns()
    _install_tts_hooks(total)
    sys.argv[0] = "podcastfy.client"
    runpy.run_module("podcastfy.client", run_name="__main__")


if __name__ == "__main__":
    main()
