#!/usr/bin/env python3
"""
Notes -> two-host podcast.
Script: Ollama / DeepSeek (OpenAI-compatible). Voices: Kokoro-FastAPI. Audio: ffmpeg.

Setup:
    sudo apt update && sudo apt install -y ffmpeg
    python3 -m venv ~/podcast-venv && source ~/podcast-venv/bin/activate
    pip install openai requests

Usage:
    python podcast_maker.py notes.txt
    python podcast_maker.py notes.txt --script-only     # review the dialogue first
    python podcast_maker.py notes.txt --merge           # also build one full mp3

Resumable: scripts and audio are cached, so re-running continues where it stopped.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from openai import OpenAI

VOICES = {"Alex": "am_adam", "Maya": "af_heart"}
SPEAKERS = list(VOICES)

# Edit freely. Regex on the left, spoken form on the right (applied before TTS).
PRONUNCIATIONS = {
    r"\bRBAC\b": "R back",
    r"\bACR\b": "A C R",
    r"\bACI\b": "A C I",
    r"\bNSG\b": "N S G",
    r"\bASG\b": "A S G",
    r"\bUDR\b": "U D R",
    r"\bSSPR\b": "S S P R",
    r"\bKQL\b": "K Q L",
    r"\bAKS\b": "A K S",
    r"\bVNet\b": "V net",
}

SYSTEM_PROMPT = """You turn technical study notes into a natural two-host podcast between Alex and Maya.
Alex explains the concepts clearly. Maya asks sharp questions, checks understanding, and gives real-world analogies.

Rules:
- Cover EVERY fact in the notes: names, numbers, limits, commands, role names and exceptions. Never drop details to save space. Expand, do not summarize.
- Keep every memory hook and rule of thumb from the notes, and have one host repeat it at the end of its topic.
- Spoken style, short turns of 2 to 5 sentences. Brief natural reactions are fine.
- If a "Conversation so far" block is given, continue naturally from it. Do not greet again and do not repeat earlier lines.
- No markdown, bullet points, stage directions or emojis. Write symbols as words.
- Output ONLY valid JSON: {"dialogue": [{"speaker": "Alex", "text": "..."}, {"speaker": "Maya", "text": "..."}]}"""


# ---------- helpers ----------
def run(cmd):
    subprocess.run(cmd, check=True)


def clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", "", text)          # stage directions
    text = re.sub(r"[*_`#>]+", "", text)            # leftover markdown
    text = text.replace("&", " and ").replace("%", " percent")
    return re.sub(r"\s+", " ", text).strip()


def pron(text: str) -> str:
    for pattern, spoken in PRONUNCIATIONS.items():
        text = re.sub(pattern, spoken, text)
    return text


def strip_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    return re.sub(r"\s*```$", "", raw)


def split_sentences(text: str, limit: int = 350) -> list[str]:
    """Keep each TTS request short; long inputs get cut or sound rushed."""
    chunks, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(s) + 1 > limit:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------- 1. split notes into chunks the model can handle ----------
HEADING = re.compile(r"(?m)^(?=CHAPTER\s|PART\s|#{1,3}\s)")


def split_chunks(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"(?m)^[-=_]{8,}\s*$", "", text)   # dashed divider lines
    pieces = []
    for block in (b.strip() for b in HEADING.split(text)):
        if not block:
            continue
        if len(block) <= max_chars:
            pieces.append(block)
            continue
        cur = ""
        for para in re.split(r"\n\s*\n", block):
            if cur and len(cur) + len(para) > max_chars:
                pieces.append(cur)
                cur = para
            else:
                cur = f"{cur}\n\n{para}".strip()
        if cur:
            pieces.append(cur)

    chunks, cur = [], ""
    for p in pieces:                                   # pack small pieces together
        if cur and len(cur) + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------- 2. script generation ----------
def normalize(dialogue: list[dict]) -> list[dict]:
    turns = []
    for i, t in enumerate(dialogue):
        text = clean(str(t.get("text", "")))
        if not text:
            continue
        spk = t.get("speaker")
        turns.append({"speaker": spk if spk in VOICES else SPEAKERS[i % 2], "text": text})
    return turns


def llm_turns(client, args, chunk, idx, total, prev):
    ctx = ""
    if prev:
        ctx = "Conversation so far (last lines, do not repeat):\n" + "\n".join(
            f"{t['speaker']}: {t['text']}" for t in prev[-4:]) + "\n\n"
    hint = f"This is part {idx} of {total}."
    if idx == 1:
        hint += " Open with a short welcome."
    if idx == total:
        hint += " This is the final part: end with a short recap and goodbye."
    user = f"{ctx}{hint}\n\nNotes for this part:\n{chunk}"

    need = int(len(chunk.split()) * args.min_ratio)
    best = None
    for attempt in range(1, args.retries + 1):
        try:
            resp = client.chat.completions.create(
                model=args.model,
                temperature=0.7,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
            )
            data = json.loads(strip_fences(resp.choices[0].message.content))
            turns = normalize(data["dialogue"])
            if len(turns) < 2:
                raise ValueError("too few turns")
            words = sum(len(t["text"].split()) for t in turns)
            if best is None or words > best[0]:
                best = (words, turns)
            if words >= need:
                return turns
            print(f"  part {idx}: {words}/{need} words, content may be dropped, retry {attempt}")
        except Exception as e:
            print(f"  part {idx}: attempt {attempt} failed: {e}")
            time.sleep(2)
    if best:
        print(f"  part {idx}: WARNING using shorter script ({best[0]} words). Check it.")
        return best[1]
    raise RuntimeError(f"Could not generate script for part {idx}")


# ---------- 3. audio ----------
def tts(text, voice, path: Path, args):
    if path.exists() and path.stat().st_size > 0:
        return
    for attempt in range(1, 5):
        try:
            r = requests.post(
                args.tts_url,
                json={"model": "kokoro", "input": text, "voice": voice,
                      "response_format": "mp3", "speed": args.speed},
                timeout=120,
            )
            r.raise_for_status()
            path.write_bytes(r.content)
            return
        except requests.RequestException as e:
            print(f"  TTS attempt {attempt} failed: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"TTS failed for: {text[:60]}")


def ensure_silence(path: Path, ms: int = 350) -> Path:
    if not path.exists():
        run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
             "anullsrc=r=24000:cl=mono", "-t", str(ms / 1000),
             "-c:a", "libmp3lame", str(path)])
    return path


def ffmpeg_concat(files, out: Path, reencode: bool):
    lst = out.with_suffix(".txt")
    lst.write_text("".join(f"file '{Path(f).resolve()}'\n" for f in files), encoding="utf-8")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if reencode:   # also evens out loudness between the two voices
        cmd += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "libmp3lame", "-b:a", "96k"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(out))
    run(cmd)
    lst.unlink()


def render_section(turns, out_mp3: Path, work: Path, args):
    work.mkdir(parents=True, exist_ok=True)
    silence = ensure_silence(work / "silence.mp3")
    files = []
    for n, t in enumerate(turns, 1):
        print(f"  [{n}/{len(turns)}] {t['speaker']}: {t['text'][:50]}...")
        voice = VOICES[t["speaker"]]
        for piece in split_sentences(pron(t["text"])):
            key = hashlib.sha1(f"{voice}|{args.speed}|{piece}".encode()).hexdigest()[:16]
            f = work / f"{key}.mp3"
            tts(piece, voice, f, args)
            files.append(f)
        files.append(silence)
    ffmpeg_concat(files, out_mp3, reencode=True)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Study notes -> two-host podcast")
    ap.add_argument("notes")
    ap.add_argument("--out-dir")
    ap.add_argument("--llm-url", default=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "gemma2:9b"))
    ap.add_argument("--tts-url", default=os.getenv("KOKORO_URL", "http://localhost:8880/v1/audio/speech"))
    ap.add_argument("--max-chars", type=int, default=4000, help="notes characters per LLM call")
    ap.add_argument("--min-ratio", type=float, default=0.8, help="min dialogue/notes word ratio")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--script-only", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    notes_path = Path(args.notes)
    out_dir = Path(args.out_dir) if args.out_dir else notes_path.parent / f"{notes_path.stem}_podcast"
    (out_dir / "scripts").mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"

    chunks = split_chunks(notes_path.read_text(encoding="utf-8"), args.max_chars)
    print(f"{len(chunks)} parts from {notes_path.name}")

    client = OpenAI(base_url=args.llm_url, api_key=os.getenv("LLM_API_KEY", "ollama"))
    prev, section_mp3s = [], []

    for i, chunk in enumerate(chunks, 1):
        script_file = out_dir / "scripts" / f"{i:02d}.json"
        if script_file.exists():
            turns = json.loads(script_file.read_text(encoding="utf-8"))
        else:
            print(f"Script {i}/{len(chunks)}")
            turns = llm_turns(client, args, chunk, i, len(chunks), prev)
            script_file.write_text(json.dumps(turns, indent=2, ensure_ascii=False), encoding="utf-8")
        prev = turns
        if args.script_only:
            continue

        mp3 = out_dir / f"{notes_path.stem}_{i:02d}.mp3"
        if not mp3.exists():
            print(f"Audio {i}/{len(chunks)}")
            render_section(turns, mp3, work, args)
        section_mp3s.append(mp3)

    if args.merge and section_mp3s:
        full = out_dir / f"{notes_path.stem}_full.mp3"
        ffmpeg_concat(section_mp3s, full, reencode=False)
        print(f"Merged: {full}")
    print(f"Done. Output in {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped. Run again to resume.")
