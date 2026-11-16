"""Edge TTS provider implementation."""

import asyncio
import os
import re
import tempfile
from typing import List
import edge_tts
from ..base import TTSProvider


class EdgeTTS(TTSProvider):
    def __init__(self, api_key: str = None, model: str = None):
        self.model = model or "default"

    def generate_audio(self, text: str, voice: str, model: str, voice2: str = None) -> bytes:
        """Generate audio using Edge TTS with SSML stripping and retries."""
        import nest_asyncio
        nest_asyncio.apply()

        # 1. Strip any SSML/XML tags Podcastfy injects (edge-tts >=7 rejects raw SSML)
        cleaned = re.sub(r"<[^>]+>", " ", text or "")
        # 2. Normalize em/en dashes and strip wrapping quotes/whitespace
        cleaned = cleaned.replace("—", ", ").replace("–", ", ")
        cleaned = re.sub(r'^["\'\s]+|["\'\s]+$', "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if not cleaned or not re.search(r"[A-Za-z0-9]", cleaned):
            return b""

        async def _try_voice(target_voice: str, max_attempts: int) -> bytes:
            last_err = None
            for attempt in range(1, max_attempts + 1):
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                    temp_path = tmp_file.name
                try:
                    communicate = edge_tts.Communicate(cleaned, target_voice)
                    await communicate.save(temp_path)
                    with open(temp_path, "rb") as f:
                        data = f.read()
                    if data:
                        await asyncio.sleep(0.5)
                        return data
                except Exception as exc:
                    last_err = exc
                    print(f"[EdgeTTS retry {attempt}/{max_attempts}] voice={target_voice} text={cleaned[:60]!r} err={exc}")
                    await asyncio.sleep(1.5 * attempt)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            raise last_err or RuntimeError(f"Edge TTS failed on: {cleaned[:80]!r}")

        async def _generate() -> bytes:
            try:
                return await _try_voice(voice, 4)
            except Exception as primary_err:
                fallbacks = (
                    ["en-US-BrianNeural", "en-US-GuyNeural"]
                    if any(k in (voice or "") for k in ("Andrew", "Brian", "Guy", "Ryan", "William"))
                    else ["en-US-JennyNeural", "en-US-EmmaNeural"]
                )
                for fb in fallbacks:
                    if fb == voice:
                        continue
                    try:
                        print(f"[EdgeTTS fallback] {voice} failed, trying {fb}")
                        return await _try_voice(fb, 2)
                    except Exception:
                        continue
                raise primary_err

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_generate())

    def get_supported_tags(self) -> List[str]:
        return []
