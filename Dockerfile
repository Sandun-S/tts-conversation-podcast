FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Podcastfy with pinned dependencies plus FastAPI/Uvicorn/pypdf for the Web UI.
RUN pip install --upgrade pip && \
    pip install \
      podcastfy \
      playwright \
      "click==8.1.8" \
      "langchain==0.3.21" \
      "langchain-community==0.3.20" \
      edge-tts \
      nest_asyncio \
      fastapi \
      "uvicorn[standard]" \
      python-multipart \
      pypdf \
      pyyaml \
      requests

# Apply the EdgeTTS SSML-stripping and retry patch directly into site-packages
COPY app/edge_tts_patch.py /tmp/edge_tts_patch.py
RUN python -c "import podcastfy.tts.providers.edge as e, shutil; shutil.copy('/tmp/edge_tts_patch.py', e.__file__); print('Patched:', e.__file__)" && \
    rm -f /tmp/edge_tts_patch.py

COPY app/ /app/app/

RUN mkdir -p /app/data/audio/tmp /app/data/transcripts /app/docs && touch /app/.env

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
