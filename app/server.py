import asyncio
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pypdf import PdfReader

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(os.getenv("APP_ROOT", "/app"))
DATA_DIR = ROOT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
TMP_DIR = DATA_DIR / "tmp_jobs"
DEFAULT_CONFIG_PATH = ROOT_DIR / "conversation_config.yaml"

for d in (AUDIO_DIR, AUDIO_DIR / "tmp", TRANSCRIPTS_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Podcastfy Studio UI")

VOICE_PRESETS = {
    "edge": [
        "en-US-AvaNeural",
        "en-US-AndrewNeural",
        "en-US-EmmaNeural",
        "en-US-BrianNeural",
        "en-US-JennyNeural",
        "en-US-GuyNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
        "en-AU-NatashaNeural",
        "en-AU-WilliamNeural",
    ],
    "kokoro": [
        "af_heart",
        "af_bella",
        "af_sarah",
        "af_nicole",
        "af_sky",
        "af_nova",
        "af_river",
        "am_adam",
        "am_michael",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_onyx",
        "bf_emma",
        "bf_isabella",
        "bm_george",
        "bm_lewis",
    ],
}

FALLBACK_CONFIG: Dict[str, Any] = {
    "conversation_style": ["engaging", "technical", "clear"],
    "roles_person1": "Senior Cloud and Linux Architect",
    "roles_person2": "Curious Engineer Preparing for Certification",
    "dialogue_structure": [
        "Introduction and Big Picture",
        "Deep Dive into Core Concepts",
        "Common Exam Traps and Edge Cases",
        "Rapid-Fire Summary",
    ],
    "podcast_name": "Study Deep Dive",
    "podcast_tagline": "Technical notes turned into a two-host discussion",
    "output_language": "English",
    "engagement_techniques": [
        "real-world analogies",
        "clarifying questions",
        "practical examples",
    ],
    "creativity": 0.6,
    "ending_message": "Catch you in the next study module!",
    "audio_format": "mp3",
    "edge_voice_1": "en-US-AvaNeural",
    "edge_voice_2": "en-US-BrianNeural",
    "kokoro_voice_1": "af_heart",
    "kokoro_voice_2": "am_adam",
    "tts_provider": "edge",
    "llm_model": "deepseek/deepseek-chat",
    "kokoro_url": os.getenv("OPENAI_BASE_URL", "http://kokoro-tts:8880/v1"),
}

JOBS: Dict[str, Dict[str, Any]] = {}
JOB_PROCS: Dict[str, asyncio.subprocess.Process] = {}


def load_defaults() -> Dict[str, Any]:
    cfg = dict(FALLBACK_CONFIG)
    if DEFAULT_CONFIG_PATH.exists():
        try:
            raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
            for key in (
                "conversation_style",
                "roles_person1",
                "roles_person2",
                "dialogue_structure",
                "podcast_name",
                "podcast_tagline",
                "output_language",
                "engagement_techniques",
                "creativity",
            ):
                if key in raw:
                    cfg[key] = raw[key]
            tts = raw.get("text_to_speech", {})
            if "ending_message" in tts:
                cfg["ending_message"] = tts["ending_message"]
            if "audio_format" in tts:
                cfg["audio_format"] = tts["audio_format"]
            edge_v = tts.get("edge", {}).get("default_voices", {})
            if edge_v.get("question"):
                cfg["edge_voice_1"] = edge_v["question"]
            if edge_v.get("answer"):
                cfg["edge_voice_2"] = edge_v["answer"]
            kokoro_v = tts.get("openai", {}).get("default_voices", {})
            if kokoro_v.get("question"):
                cfg["kokoro_voice_1"] = kokoro_v["question"]
            if kokoro_v.get("answer"):
                cfg["kokoro_voice_2"] = kokoro_v["answer"]
        except Exception:
            pass
    return cfg


def extract_text_from_upload(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        tmp_pdf = TMP_DIR / f"upload_{uuid.uuid4().hex}.pdf"
        tmp_pdf.write_bytes(content)
        try:
            reader = PdfReader(str(tmp_pdf))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(p.strip() for p in pages if p.strip())
        finally:
            tmp_pdf.unlink(missing_ok=True)
    return content.decode("utf-8", errors="replace")


def parse_csv_or_lines(value: str) -> List[str]:
    items = [x.strip() for x in re.split(r"[\n,]+", value or "") if x.strip()]
    return items


def snapshot_files(directory: Path) -> Dict[str, float]:
    if not directory.exists():
        return {}
    return {
        p.name: p.stat().st_mtime
        for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".")
    }


def diff_new_files(before: Dict[str, float], after: Dict[str, float]) -> List[str]:
    changed = [
        name
        for name, mtime in after.items()
        if name not in before or mtime > before[name]
    ]
    changed.sort(key=lambda n: after[n], reverse=True)
    return changed


@app.get("/", response_class=HTMLResponse)
def index():
    return (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/defaults")
def get_defaults():
    return {
        "config": load_defaults(),
        "voices": VOICE_PRESETS,
        "has_env_deepseek": bool(os.getenv("DEEPSEEK_API_KEY")),
    }


@app.get("/api/kokoro-voices")
def get_kokoro_voices(url: str = "http://kokoro-tts:8880/v1"):
    base = url.rstrip("/")
    try:
        r = requests.get(f"{base}/audio/voices", timeout=3)
        r.raise_for_status()
        data = r.json()
        voices = data.get("voices") if isinstance(data, dict) else data
        if isinstance(voices, list) and voices:
            return {"reachable": True, "voices": sorted(set(str(v) for v in voices))}
    except Exception as exc:
        return {"reachable": False, "voices": VOICE_PRESETS["kokoro"], "error": str(exc)}
    return {"reachable": True, "voices": VOICE_PRESETS["kokoro"]}


@app.get("/api/files")
def list_files():
    def _items(folder: Path, kind: str) -> List[Dict[str, Any]]:
        if not folder.exists():
            return []
        files = [p for p in folder.iterdir() if p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [
            {
                "name": p.name,
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
                "url": f"/api/download/{kind}/{p.name}",
            }
            for p in files[:50]
        ]

    return {
        "audio": _items(AUDIO_DIR, "audio"),
        "transcripts": _items(TRANSCRIPTS_DIR, "transcripts"),
    }


@app.get("/api/download/{kind}/{filename}")
def download_file(kind: str, filename: str):
    base = {"audio": AUDIO_DIR, "transcripts": TRANSCRIPTS_DIR}.get(kind)
    if base is None:
        raise HTTPException(status_code=404, detail="Invalid file category")
    safe_name = Path(filename).name
    target = (base / safe_name).resolve()
    if not str(target).startswith(str(base.resolve())) or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target, filename=safe_name)


async def run_podcast_job(
    job_id: str,
    cmd: List[str],
    env: Dict[str, str],
    input_path: Path,
    config_path: Path,
):
    job = JOBS[job_id]
    before_audio = snapshot_files(AUDIO_DIR)
    before_tx = snapshot_files(TRANSCRIPTS_DIR)

    job["logs"].append(f"$ {' '.join(cmd)}")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=str(ROOT_DIR),
        )
        JOB_PROCS[job_id] = proc
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                job["logs"].append(text)
                if len(job["logs"]) > 500:
                    job["logs"] = job["logs"][-500:]

        rc = await proc.wait()
        after_audio = snapshot_files(AUDIO_DIR)
        after_tx = snapshot_files(TRANSCRIPTS_DIR)
        job["new_audio"] = diff_new_files(before_audio, after_audio)
        job["new_transcripts"] = diff_new_files(before_tx, after_tx)
        if job.get("cancelled"):
            job["status"] = "cancelled"
        else:
            job["status"] = "completed" if rc == 0 else "failed"
        job["exit_code"] = rc
        job["finished_at"] = time.time()
    except Exception as exc:
        job["status"] = "failed"
        job["logs"].append(f"ERROR: {exc}")
        job["finished_at"] = time.time()
    finally:
        JOB_PROCS.pop(job_id, None)
        input_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)


@app.post("/api/generate")
async def generate_podcast(
    mode: str = Form("full"),
    tts_provider: str = Form("edge"),
    voice_1: str = Form("en-US-AvaNeural"),
    voice_2: str = Form("en-US-AndrewNeural"),
    kokoro_url: str = Form("http://kokoro-tts:8880/v1"),
    llm_model: str = Form("deepseek/deepseek-chat"),
    api_key: str = Form(""),
    api_key_label: str = Form("DEEPSEEK_API_KEY"),
    text_input: str = Form(""),
    podcast_name: str = Form("Study Deep Dive"),
    podcast_tagline: str = Form("Technical notes turned into a two-host discussion"),
    roles_person1: str = Form("Senior Cloud and Linux Architect"),
    roles_person2: str = Form("Curious Engineer Preparing for Certification"),
    conversation_style: str = Form("engaging, technical, clear"),
    dialogue_structure: str = Form(
        "Introduction and Big Picture\nDeep Dive into Core Concepts\nCommon Exam Traps and Edge Cases\nRapid-Fire Summary"
    ),
    engagement_techniques: str = Form("real-world analogies, clarifying questions, practical examples"),
    output_language: str = Form("English"),
    creativity: float = Form(0.6),
    ending_message: str = Form("Catch you in the next study module!"),
    audio_format: str = Form("mp3"),
    file: Optional[UploadFile] = File(None),
):
    parts: List[str] = []
    if file is not None and file.filename:
        raw_bytes = await file.read()
        if raw_bytes:
            extracted = extract_text_from_upload(file.filename, raw_bytes)
            if extracted.strip():
                parts.append(extracted.strip())
    if text_input.strip():
        parts.append(text_input.strip())

    combined_text = "\n\n".join(parts).strip()
    if not combined_text:
        raise HTTPException(status_code=400, detail="Provide a file (.txt/.md/.pdf) or paste text.")

    effective_key = api_key.strip() or os.getenv(api_key_label.strip(), "")
    if mode == "full" and not effective_key:
        raise HTTPException(
            status_code=400,
            detail=f"API key ({api_key_label}) is required for Full Script + Audio mode. Switch to Transcript-Only mode to skip LLM.",
        )

    job_id = uuid.uuid4().hex[:12]
    input_path = TMP_DIR / f"input_{job_id}.txt"
    config_path = TMP_DIR / f"config_{job_id}.yaml"
    input_path.write_text(combined_text, encoding="utf-8")

    podcastfy_tts_model = "openai" if tts_provider == "kokoro" else "edge"
    cfg_dict = {
        "conversation_style": parse_csv_or_lines(conversation_style) or ["engaging", "clear"],
        "roles_person1": roles_person1.strip() or "Host 1",
        "roles_person2": roles_person2.strip() or "Host 2",
        "dialogue_structure": parse_csv_or_lines(dialogue_structure)
        or ["Introduction", "Main Discussion", "Summary"],
        "podcast_name": podcast_name.strip() or "Study Deep Dive",
        "podcast_tagline": podcast_tagline.strip() or "",
        "output_language": output_language.strip() or "English",
        "engagement_techniques": parse_csv_or_lines(engagement_techniques) or ["examples"],
        "creativity": float(creativity),
        "text_to_speech": {
            "default_tts_model": podcastfy_tts_model,
            "output_directories": {
                "transcripts": str(TRANSCRIPTS_DIR),
                "audio": str(AUDIO_DIR),
            },
            "edge": {
                "default_voices": {
                    "question": voice_1.strip(),
                    "answer": voice_2.strip(),
                }
            },
            "openai": {
                "default_voices": {
                    "question": voice_1.strip(),
                    "answer": voice_2.strip(),
                },
                "model": "kokoro",
            },
            "audio_format": audio_format.strip() or "mp3",
            "temp_audio_dir": str(AUDIO_DIR / "tmp") + "/",
            "ending_message": ending_message.strip(),
        },
    }
    config_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    if tts_provider == "kokoro":
        clean_kokoro = kokoro_url.strip().rstrip("/")
        env["OPENAI_BASE_URL"] = clean_kokoro
        env["OPENAI_API_BASE"] = clean_kokoro
        env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or "not-needed"

    cmd = [
        "python",
        "-m",
        "app.tts_progress_runner",
        "--tts-model",
        podcastfy_tts_model,
        "--conversation-config",
        str(config_path),
    ]
    if mode == "transcript_only":
        cmd += ["--transcript", str(input_path)]
    else:
        key_label = api_key_label.strip() or "DEEPSEEK_API_KEY"
        env[key_label] = effective_key
        cmd += [
            "--file",
            str(input_path),
            "--llm-model-name",
            llm_model.strip() or "deepseek/deepseek-chat",
            "--api-key-label",
            key_label,
        ]

    JOBS[job_id] = {
        "id": job_id,
        "status": "running",
        "mode": mode,
        "tts_provider": tts_provider,
        "logs": [],
        "new_audio": [],
        "new_transcripts": [],
        "started_at": time.time(),
        "finished_at": None,
    }

    asyncio.create_task(run_podcast_job(job_id, cmd, env, input_path, config_path))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    proc = JOB_PROCS.get(job_id)
    if proc and proc.returncode is None:
        job["cancelled"] = True
        job["logs"].append("[Cancelled by user]")
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except asyncio.TimeoutError:
            proc.kill()
        return {"status": "cancelled"}
    return {"status": job["status"]}
