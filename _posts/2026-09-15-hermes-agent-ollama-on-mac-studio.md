---
layout:     post
title:      Run Hermes Agent with Colima, Ollama, and Metal on a Mac Studio
subtitle:   A local-first deployment with CPU tools in Colima containers and GPU inference on the host
date:       2026-09-15
author:     feizaipp
header-img: img/post-bg-desk.jpg
catalog: true
tags:
    - Hermes Agent
    - Ollama
    - Colima
    - Docker
    - Apple Silicon
    - Local LLM
---

> This guide is for an Apple Silicon Mac Studio with 128 GB unified memory. It keeps model inference and other Metal-accelerated workloads on macOS, while Hermes Agent and its ordinary CPU tools run in Colima containers.

## 1. What we are building

This deployment deliberately separates performance-sensitive work from untrusted agent work:

```text
macOS host
├── Ollama + local models                # Metal / unified memory
├── Optional MLX, whisper.cpp, ComfyUI   # Metal / host-only GPU tools
└── Colima (Docker-compatible Linux VM)
    └── Hermes Agent container
        ├── Hermes gateway and dashboard
        ├── CPU tools: Python, Node, Git, PDF/image/media utilities
        └── /workspace                    # only project folder intentionally shared with the agent
```

Do **not** run Ollama inside Colima on a Mac. Colima runs Linux containers inside a VM and does not provide the normal native Metal execution path. Keep Ollama, MLX, `llama.cpp` compiled with Metal, and other GPU tools on macOS.

The Hermes container can access host Ollama at `host.docker.internal:11434`. This adds negligible local-network overhead; model loading and inference still happen on the Mac GPU/unified memory.

This guide also intentionally does **not** mount the Docker socket into Hermes. A process that can control the Docker daemon can usually escape the practical security boundary of its own container.

## 2. Prerequisites

Before starting, make sure that you have:

* An Apple Silicon Mac running a current macOS release.
* At least 120 GB of free SSD space. More is better if you plan to keep several models.
* Colima, Docker CLI, and Docker Compose. Colima is a lightweight, Docker-compatible runtime for macOS.
* Homebrew. Install it from the [official Homebrew site](https://brew.sh/) if it is not already installed.

Install the lightweight container runtime and a few host-side utilities. `uv` is useful later for host-only GPU tools; it is not used for normal agent tool execution.

```bash
brew install colima docker docker-compose git uv

# A good starting allocation for Hermes and CPU tools. Ollama remains outside
# this VM and uses the Mac Studio's unified memory directly.
colima start --cpu 6 --memory 12 --disk 80 --vm-type vz --mount-type virtiofs

docker run --rm hello-world
docker compose version
```

The `vz` and `virtiofs` options use Apple's virtualization and fast file-sharing facilities on current macOS releases. If your macOS version does not support them, run `colima start` without those two options. Colima creates a Docker-compatible context automatically, so every `docker compose` command in the rest of this guide is served by Colima.

Install the macOS version of Ollama from the [official download page](https://ollama.com/download/mac), open it once, and then verify its local API:

```bash
curl http://127.0.0.1:11434/api/tags
```

You should receive JSON, even if the model list is initially empty. Do not change Ollama to listen on your LAN. The default local-only listener is the desired setting.

## 3. Download the first models

For a 128 GB Mac Studio, start with one general model and one coding model instead of downloading everything at once:

```bash
ollama pull qwen3.6
ollama pull qwen3-coder:30b
```

Use `qwen3.6` as the initial general-purpose model. Switch Hermes to `qwen3-coder:30b` when working on code repositories. The 480B Qwen3-Coder model is not a sensible local target for 128 GB unified memory; its Ollama page lists a 250 GB minimum for local execution.

Hermes needs a sufficiently large context window for its system prompt, tool schemas, and multi-step history. Create two local model aliases with a persistent 64K context setting:

```bash
cat > Modelfile.qwen3.6-hermes <<'EOF'
FROM qwen3.6
PARAMETER num_ctx 65536
EOF

cat > Modelfile.qwen3-coder-hermes <<'EOF'
FROM qwen3-coder:30b
PARAMETER num_ctx 65536
EOF

ollama create qwen3.6-hermes -f Modelfile.qwen3.6-hermes
ollama create qwen3-coder-hermes -f Modelfile.qwen3-coder-hermes
```

Then run:

```bash
ollama run qwen3.6-hermes "Reply with exactly: Ollama is ready."
ollama ps
```

The `CONTEXT` column should show at least `65536`. The `num_ctx` setting is now carried by the two local aliases, so it remains in effect after restarting the Ollama app. Do not continue until `ollama ps` confirms the intended context size.

## 4. Create the deployment directory

The commands below create three clearly separated locations:

* `~/Services/hermes` holds the Compose files and CPU-tool image definition.
* `~/.hermes` persists Hermes configuration, skills, sessions, and memory.
* `~/HermesWorkspace` is the only project directory shared with the agent. Hermes also needs write access to its own state in `~/.hermes`; do not treat that state directory as a general workspace.

```bash
mkdir -p ~/Services/hermes
mkdir -p ~/.hermes
mkdir -p ~/HermesWorkspace
cd ~/Services/hermes
```

Create `Dockerfile` with the following contents:

```dockerfile
FROM nousresearch/hermes-agent:latest

USER root

# CPU-only command-line tools available to Hermes inside its container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       git \
       imagemagick \
       jq \
       pandoc \
       poppler-utils \
       ripgrep \
       sqlite3 \
       yq \
    && rm -rf /var/lib/apt/lists/*

# agent-browser is optional, but installing it now avoids the first-use download.
RUN npm install -g agent-browser

USER hermes
```

These are deliberately CPU-oriented utilities: source control, code search, JSON/YAML processing, document conversion, PDF extraction, image conversion, media handling, and SQLite. Add a package only when you know why Hermes needs it; a smaller image has a smaller maintenance and attack surface.

Now create `compose.yaml`:

```yaml
services:
  hermes:
    build: .
    image: local/hermes-cpu:latest
    container_name: hermes
    restart: unless-stopped
    command: ["gateway", "run"]
    environment:
      HERMES_UID: "${HERMES_UID}"
      HERMES_GID: "${HERMES_GID}"
    env_file:
      - ${HOME}/.hermes/.env
    volumes:
      - ${HOME}/.hermes:/opt/data
      - ${HOME}/HermesWorkspace:/workspace

  dashboard:
    image: local/hermes-cpu:latest
    container_name: hermes-dashboard
    restart: unless-stopped
    depends_on:
      - hermes
    command: ["dashboard", "--host", "0.0.0.0", "--port", "9119", "--no-open"]
    environment:
      HERMES_UID: "${HERMES_UID}"
      HERMES_GID: "${HERMES_GID}"
    ports:
      - "127.0.0.1:9119:9119"
    volumes:
      - ${HOME}/.hermes:/opt/data
```

Create `.env` in the same directory. This file supplies your macOS user and group IDs so that container-created files remain readable on the host:

```bash
printf 'HERMES_UID=%s\nHERMES_GID=%s\n' "$(id -u)" "$(id -g)" > .env
```

The dashboard is deliberately published only to `127.0.0.1`. Do not replace this with `0.0.0.0`. If you later need remote access, use Tailscale or an authenticated reverse proxy rather than exposing the dashboard directly.

## 5. Build the image and configure Hermes

Build the local image first:

```bash
docker compose build
```

Configure Hermes through its interactive model wizard. It writes configuration into the persistent `~/.hermes` directory, not into the disposable container filesystem:

```bash
docker compose run --rm hermes model
```

Choose **Custom endpoint (self-hosted / vLLM / etc.)** and enter the following values:

```text
API base URL: http://host.docker.internal:11434/v1
API key: ollama
Model name: qwen3.6-hermes
Context length: 65536
```

`ollama` is only a placeholder API-key value for the OpenAI-compatible client. A standard local Ollama endpoint does not validate it.

Next, inspect `~/.hermes/config.yaml` and ensure it contains these important settings. Add missing fields; do not remove settings created by the wizard.

```yaml
model:
  default: qwen3.6-hermes
  provider: custom
  base_url: http://host.docker.internal:11434/v1
  api_key: ollama
  context_length: 65536

# "local" here means the local process namespace of the Hermes container.
# Normal terminal and file tools therefore stay inside the Colima container.
terminal:
  backend: local
  cwd: /workspace

approvals:
  mode: manual

skills:
  write_approval: true

memory:
  write_approval: true
```

Do not set `terminal.backend: docker` in this particular design. Hermes is already inside the Colima security container; selecting a nested Docker backend would require access to a Docker daemon, which is exactly the high-privilege Docker socket we intentionally did not mount.

Verify that the container can see Ollama before starting the service:

```bash
docker compose run --rm hermes curl -s http://host.docker.internal:11434/v1/models
```

If this does not return a JSON model list, stop here and verify that Ollama is running on the host. In Colima, `host.docker.internal` resolves back to the macOS host; `localhost` would refer to the Hermes container itself.

## 6. Start Hermes and validate the installation

Start both long-running services:

```bash
docker compose up -d
docker compose ps
docker compose logs --tail=100 hermes
```

Open the local dashboard at [http://127.0.0.1:9119](http://127.0.0.1:9119). It should be reachable only from this Mac.

For a terminal chat, use:

```bash
docker compose run --rm -it hermes chat
```

Run these three checks in Hermes:

1. Ask: `What model are you using?`
2. Ask: `Create hello.txt in /workspace containing one short sentence, then read it back.`
3. Ask: `Run rg --version and jq --version, then report the results.`

The second check proves that Hermes can work in the intended shared workspace. The third confirms that your CPU tools are present in the container.

For browser automation, enable the browser toolset through `hermes tools` or the dashboard. The preinstalled `agent-browser` avoids an on-demand download, but it is still a CPU/browser workload inside Colima.

## 7. Switching to the coding model

Keep the normal model as the default, and switch to the coding model for repository work:

```bash
docker compose run --rm -it hermes model
```

Choose the same endpoint and enter:

```text
Model name: qwen3-coder-hermes
Context length: 65536
```

You can change it back at any time. Before starting a long coding task, use `ollama ps` on the host to confirm that only the model you intend to use is loaded. Running several large models at once wastes unified memory and can make response latency unpredictable.

## 8. Host-only GPU tools

Ollama is already a host-only GPU tool and Hermes reaches it through its local HTTP API. Treat additional GPU tools the same way:

* Install MLX Python packages, `llama.cpp` with Metal, `whisper.cpp`, or image-generation software on macOS, not in the Hermes image.
* Run each tool as a small host-side service bound to `127.0.0.1`.
* Give it a narrow API that exposes only the action Hermes needs, such as `transcribe`, `embed`, or `generate_image`.
* Require a bearer token on that API, store the token in `~/.hermes/.env`, and never expose the service to the LAN.
* Call it from Hermes through `http://host.docker.internal:<port>`.

Avoid giving Hermes a host shell, SSH key, Docker socket, or a broad `~/` volume mount just to access a GPU tool. A small allowlisted API keeps Metal performance while preserving the reason the container exists.

### 8.1 A complete Whisper example: host Metal, container client

This section installs `whisper.cpp` on macOS and exposes one narrow transcription endpoint to Hermes. The endpoint accepts only a file name from `~/HermesWorkspace/audio`, never a host command or an arbitrary host path. The same folder appears as `/workspace/audio` inside Hermes.

The official `whisper.cpp` project supports Apple Silicon Metal inference and includes both `whisper-cli` and `whisper-server`. We use `whisper-cli` behind a small authenticated wrapper because the wrapper can enforce an audio-directory allowlist and never exposes the server's model-loading endpoint.

#### Step 1: Install Whisper and download a model on macOS

Run these commands on the **macOS host**, not inside Colima:

```bash
brew install whisper.cpp ffmpeg

mkdir -p ~/Services/whisper-bridge/models
mkdir -p ~/HermesWorkspace/audio

git clone https://github.com/ggml-org/whisper.cpp.git ~/Services/whisper-bridge/whisper.cpp
cd ~/Services/whisper-bridge/whisper.cpp
sh ./models/download-ggml-model.sh large-v3-turbo ../models
```

`large-v3-turbo` is a sensible quality/speed starting point for a Mac Studio. If you primarily transcribe English and want lower memory use, use `small.en` instead. The `whisper.cpp` model downloader supports both names.

Confirm that the binary is available and that local Metal inference works before adding Hermes:

```bash
whisper-cli -m ~/Services/whisper-bridge/models/ggml-large-v3-turbo.bin \
  -f ~/HermesWorkspace/audio/example.wav
```

For an MP3, M4A, or other common input format, use the bridge below; it converts the input safely to a 16 kHz mono WAV file before transcription.

#### Step 2: Create a local, authenticated transcription bridge

Create a random secret. It protects the local HTTP endpoint from other processes on the Mac. The first command saves a host-only copy with restrictive permissions; the second puts the same value into Hermes' persistent environment:

```bash
mkdir -p ~/Services/whisper-bridge
umask 077
openssl rand -hex 32 > ~/Services/whisper-bridge/token
printf 'WHISPER_BRIDGE_TOKEN=%s\n' "$(cat ~/Services/whisper-bridge/token)" >> ~/.hermes/.env
```

Create `~/Services/whisper-bridge/app.py`:

```python
import os
import re
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

HOME = Path.home()
AUDIO_DIR = (HOME / "HermesWorkspace" / "audio").resolve()
MODEL = HOME / "Services" / "whisper-bridge" / "models" / "ggml-large-v3-turbo.bin"
TOKEN = (HOME / "Services" / "whisper-bridge" / "token").read_text().strip()
WHISPER = os.environ.get("WHISPER_BIN", "whisper-cli")
ALLOWED_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".webm"}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class TranscriptionRequest(BaseModel):
    file: str
    language: str = "auto"


def require_token(authorization: str | None) -> None:
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


def resolve_audio(name: str) -> Path:
    # A file name only: no directories, symlinks, or arbitrary host paths.
    if Path(name).name != name or Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="unsupported audio file")
    candidate = (AUDIO_DIR / name).resolve()
    if candidate.parent != AUDIO_DIR or not candidate.is_file():
        raise HTTPException(status_code=404, detail="audio file not found")
    if candidate.stat().st_size > 2 * 1024 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio file is too large")
    return candidate


@app.get("/health")
def health():
    return {"ok": MODEL.is_file()}


@app.post("/v1/transcriptions")
def transcribe(request: TranscriptionRequest, authorization: str | None = Header(default=None)):
    require_token(authorization)
    if request.language != "auto" and not re.fullmatch(r"[a-z]{2}", request.language):
        raise HTTPException(status_code=400, detail="language must be auto or a two-letter code")

    source = resolve_audio(request.file)
    with tempfile.TemporaryDirectory(prefix="whisper-") as temp_dir:
        temp = Path(temp_dir)
        wav = temp / "input.wav"
        output_base = temp / "transcript"
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-i", str(source), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                [WHISPER, "-m", str(MODEL), "-f", str(wav), "-otxt", "-of", str(output_base), "-l", request.language],
                check=True, capture_output=True, text=True,
            )
            return {"text": Path(f"{output_base}.txt").read_text().strip(), "file": source.name}
        except subprocess.CalledProcessError as error:
            raise HTTPException(status_code=500, detail="transcription failed") from error
```

Install the small host-side Python environment and start the bridge in the foreground for its first test:

```bash
cd ~/Services/whisper-bridge
uv venv
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]"
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8790
```

In a second host Terminal, check that the service is alive:

```bash
curl http://127.0.0.1:8790/health
```

It should return `{"ok":true}`. Binding to `127.0.0.1` prevents LAN access. Colima reaches this host-only service using the special `host.docker.internal` address.

#### Step 3: Verify the call from the Hermes container

Place an audio file in `~/HermesWorkspace/audio`, then test from the Hermes container. The file must be addressed by its file name, not its path:

```bash
docker compose run --rm hermes \
  curl --fail --silent --show-error \
  -H "Authorization: Bearer $(cat ~/Services/whisper-bridge/token)" \
  -H "Content-Type: application/json" \
  -d '{"file":"meeting.m4a","language":"zh"}' \
  http://host.docker.internal:8790/v1/transcriptions
```

Replace `meeting.m4a` and `zh` with your actual file and language. Use `auto` when you do not want to force a language. A successful response contains JSON with the transcribed text.

#### Step 4: Give Hermes clear instructions for the new capability

Create a local Hermes skill at `~/.hermes/skills/local-whisper/SKILL.md`:

````markdown
---
name: local-whisper
description: Transcribe an audio file with the host's local whisper.cpp service.
---

# Local Whisper transcription

Audio files are available only in `/workspace/audio`. Never use a path outside
that directory. Call the local service with the basename only:

```sh
curl --fail --silent --show-error \
  -H "Authorization: Bearer $WHISPER_BRIDGE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file":"FILENAME","language":"auto"}' \
  http://host.docker.internal:8790/v1/transcriptions
```

The response JSON contains `text`. Report the transcript to the user. Do not
attempt to run whisper, ffmpeg, or any host command directly.
````

Restart Hermes so it reloads its skills and environment:

```bash
cd ~/Services/hermes
docker compose restart hermes
```

The `env_file` entry in `compose.yaml` makes `WHISPER_BRIDGE_TOKEN` available to Hermes' container processes. It is deliberately not added to the dashboard service, which does not need it.

#### Step 5: Start the bridge automatically after a restart

After the foreground test works, create `~/Library/LaunchAgents/io.feizaipp.whisper-bridge.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>io.feizaipp.whisper-bridge</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>exec "$HOME/Services/whisper-bridge/.venv/bin/python" -m uvicorn --app-dir "$HOME/Services/whisper-bridge" app:app --host 127.0.0.1 --port 8790</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
</dict>
</plist>
```

Load it for the current user and confirm that it survives a restart:

```bash
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/io.feizaipp.whisper-bridge.plist
launchctl print "gui/$(id -u)/io.feizaipp.whisper-bridge"
curl http://127.0.0.1:8790/health
```

At this point a request such as “Transcribe `/workspace/audio/meeting.m4a` in Chinese” will use the host's Metal-accelerated Whisper model through the restricted local API. Hermes can create or read files only within its shared workspace and cannot execute arbitrary commands on macOS.

## 9. Operating and upgrading safely

Useful daily commands:

```bash
cd ~/Services/hermes
docker compose ps
docker compose logs -f hermes
docker compose restart hermes
ollama list
ollama ps
```

To update Hermes and rebuild the CPU-tools image:

```bash
cd ~/Services/hermes
docker compose pull
docker compose build --pull
docker compose up -d --force-recreate
```

Your Hermes state remains in `~/.hermes`, so it survives container replacement. Back up that directory before major version changes; it contains configuration, credentials, session history, skills, and memory.

Finally, keep `approvals.mode: manual` while you are learning how the agent behaves. The Colima container boundary protects macOS from ordinary terminal and file activity, but it does not prevent an agent from making undesirable changes inside `/workspace` or from sending data to a service that you explicitly enable.

## 10. References

* [Colima documentation](https://colima.run/docs/)
* [Hermes Agent Docker deployment guide](https://hermes-agent.nousresearch.com/docs/user-guide/docker) (the Docker/Compose commands in this article are provided by Colima)
* [Hermes Agent: local Ollama provider](https://hermes-agent.nousresearch.com/docs/guides/local-ollama-setup)
* [Hermes Agent security configuration](https://hermes-agent.nousresearch.com/docs/user-guide/security)
* [Ollama model library: Qwen3-Coder](https://ollama.com/library/qwen3-coder)
* [whisper.cpp project and Apple Silicon guidance](https://github.com/ggml-org/whisper.cpp)
* [Homebrew whisper.cpp formula](https://formulae.brew.sh/formula/whisper.cpp)
