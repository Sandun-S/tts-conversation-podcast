# TTS Podcast Generator (Podcastfy + Kokoro TTS + DeepSeek)

Turn study notes and technical documents into two-host audio podcasts using **Podcastfy**, **DeepSeek** (`deepseek/deepseek-chat`) for script generation, and **Edge TTS** or a local **Kokoro TTS** server (`kokoro-fastapi-gpu` / `kokoro-fastapi-cpu`) via Docker.

---

## Quick Start (Docker + Web UI)

The `podcastfy-ui` service (`syntaxsan/tts_conversation_podcast-podcastfy:latest`) provides a browser UI at **`http://localhost:8000`** with:
- `.txt`, `.md`, and `.pdf` upload (or direct text paste)
- **Notes → DeepSeek → Audio** and **Transcript Only → Audio (Skip LLM)** modes
- **Edge TTS** (with SSML-stripping & retry patch) and **Kokoro TTS** switching with custom Person 1 / Person 2 voices
- Full `conversation_config.yaml` and API key configuration in the UI

Start with the profile matching your machine:

- **Edge TTS only (no GPU):**
  ```bash
  docker compose up -d
  ```
- **With Kokoro CPU (no GPU):**
  ```bash
  docker compose --profile cpu up -d
  ```
- **With Kokoro GPU (home PC with NVIDIA GPU):**
  ```bash
  docker compose --profile gpu up -d
  ```

---

## Project Structure

- `docker-compose.yml` — Runs the local `kokoro-tts` OpenAI-compatible TTS server on port `8880` with NVIDIA GPU support.
- `conversation_config.yaml` — Configures the two-host podcast style, dialogue structure, Kokoro voices (`af_heart`, `am_adam`), and output paths (`./data/transcripts`, `./data/audio`).
- `.env` — Environment variables pointing Podcastfy's OpenAI TTS client to the local Kokoro server (`http://localhost:8880/v1`).
- `docs/` — Source study notes (e.g., `docs/az104/*.txt`).
- `data/` — Generated transcripts (`data/transcripts/`) and audio files (`data/audio/`).
- `models/` — Local Kokoro TTS model files mounted into the Docker container (gitignored).

---

## Prerequisites

- **Docker** with **NVIDIA Container Toolkit** (for GPU-accelerated Kokoro TTS)
- **Python 3.12** and **[uv](https://docs.astral.sh/uv/)**
- **DeepSeek API Key**

---

## Setup

### 1. Start Kokoro TTS (Docker)

Verify GPU access in Docker (optional):

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

Start the Kokoro TTS service:

```bash
docker compose up -d
```

The OpenAI-compatible TTS endpoint will be available at `http://localhost:8880/v1`.

### 2. Python Environment

Create the virtual environment and install the pinned dependencies:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv venv ~/.venvs/podcastfy --python 3.12
source ~/.venvs/podcastfy/bin/activate

uv pip install podcastfy
uv pip install playwright
uv pip install "click==8.1.8"
uv pip install "langchain==0.3.21" "langchain-community==0.3.20"
```

Verify the CLI works:

```bash
python -m podcastfy.client --help
```

### 3. Environment Variables (`.env`)

The `.env` file configures Podcastfy to route OpenAI TTS calls to the local Kokoro server:

```env
OPENAI_BASE_URL=http://localhost:8880/v1
OPENAI_API_BASE=http://localhost:8880/v1
OPENAI_API_KEY=not-needed
```

If generating `.env` in WSL2:

```bash
WIN_IP=$(ip route show | grep -i default | awk '{print $3}')

cat << EOF > /mnt/e/Github/TTS/.env
OPENAI_BASE_URL=http://localhost:8880/v1
OPENAI_API_BASE=http://localhost:8880/v1
OPENAI_API_KEY=not-needed
OLLAMA_API_BASE=http://${WIN_IP}:11434
EOF
```

---

## Usage

Export your DeepSeek API key and run `podcastfy.client` with your notes file:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"

python -m podcastfy.client \
  --text "$(cat ./docs/az104/AZ104_Audio_Notes_Part1.txt)" \
  --llm-model-name "deepseek/deepseek-chat" \
  --api-key-label DEEPSEEK_API_KEY \
  --tts-model openai \
  --conversation-config ./conversation_config.yaml
```

Generated files are saved to:
- **Transcripts:** `./data/transcripts/`
- **Audio (`.mp3`):** `./data/audio/`
