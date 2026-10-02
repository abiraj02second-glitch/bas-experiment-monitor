# BAS AI Experiment Monitor

BAS is a browser dashboard and FastAPI backend for monitoring ordered experiment steps with computer vision, hand tracking, live alerts, logging, video upload, and automatic voice notifications.

- **Public source repository:** https://github.com/abiraj02second-glitch/bas-ai-experiment-monitor
- **Backend:** Python, FastAPI, OpenCV, YOLO, MediaPipe, and `VoiceSoundAgent`
- **Frontend:** bundled dashboard in `static/index.html`
- **Local default URL:** http://127.0.0.1:8000
- **Hosted container default port:** `3000` through the `PORT` environment variable

## Run locally on Windows

PowerShell often opens in `C:\Windows\System32`. The project commands must be run from the folder where this repository was extracted.

```powershell
cd C:\path\to\bas-ai-experiment-monitor
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000** after Uvicorn reports that it is running. If PowerShell blocks activation, run the installation and server commands without activation:

```powershell
cd C:\path\to\bas-ai-experiment-monitor
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

You can also double-click `run.bat`; it creates the virtual environment, installs dependencies, and starts the dashboard.

## Local Docker Compose

Docker is optional. It is not required for the normal Windows setup.

```powershell
cd C:\path\to\bas-ai-experiment-monitor
docker compose up --build -d
docker compose ps
```

The Compose configuration maps host port `8000` to the container port `3000`. Open **http://127.0.0.1:8000**. Stop it with:

```powershell
docker compose down
```

## Managed hosting

The project includes a production `Dockerfile` configured for managed container hosting. It listens on `PORT`, defaulting to `3000`, serves the dashboard and API from one FastAPI process, and exposes `/health` as an unauthenticated readiness endpoint. The hosted dashboard uses root-relative API paths and does not depend on `localhost`.

The hosted image intentionally uses `requirements-hosted.txt`, a lightweight runtime set. This keeps the public dashboard, simulation controls, event feed, voice agent, and video placeholder responsive during deployment. Full YOLO and MediaPipe packages remain in `requirements.txt` for local camera/model analysis; the backend loads them optionally in the background when they are installed.

Camera behavior depends on where the backend runs. A local Windows process can access the local webcam when permissions are granted. A hosted container normally cannot access a visitor's physical webcam directly; use browser-side camera capabilities or upload a video for hosted analysis. Container filesystem files such as logs, uploads, and recordings are runtime data and should not be treated as permanent storage.

## Dashboard capabilities

The dashboard provides experiment start, pause, reset, ordered step validation, next-step prompts, skipped-step and wrong-step alerts, live detection overlays, camera controls, uploaded-video analysis, timestamped activity logs, CSV export, model metrics, recording controls, resolution/FPS settings, and UDP/JPEG stream controls.

If no trained `best.pt` model is present, the backend enters Quick Start mode and uses the standard YOLO model with `steps_quickstart.json`. The demo controls can inject successful, skipped, and incorrect events so the dashboard and voice workflow can be tested without a camera.

## Automatic voice notifications

Voice output is handled in two layers:

1. The browser reads backend event messages through `speechSynthesis`, which is the audible voice output for a user's browser/device.
2. The backend `VoiceSoundAgent` uses offline `pyttsx3` speech when a server audio runtime is available.

The voice agent uses a bounded priority queue, prioritizes warnings and errors, suppresses rapid duplicates, and reports queue and error metrics in `/state.voice`. Every backend event includes a visible `text` value and a spoken `voice` value. Browser audio policies may require the user to click Start once; kiosk mode can be used for unattended browser operation.

## Backend API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/health` | Readiness and voice-agent status |
| `GET` | `/state` | Current experiment, events, log, stream, and voice state |
| `GET` | `/events?after=<id>` | Incremental event feed |
| `GET` | `/video` | MJPEG live video stream |
| `GET` | `/model` | Model metrics or fallback metrics |
| `GET` | `/log.json` | Experiment log JSON |
| `POST` | `/control/start` | Start or resume the experiment |
| `POST` | `/control/pause` | Pause the experiment |
| `POST` | `/control/reset` | Reset experiment state |
| `POST` | `/simulate/wrong` | Inject an incorrect-sequence warning |
| `POST` | `/simulate/skip` | Skip the current expected step |
| `POST` | `/voice/test` | Queue a voice test; accepts optional `{"text":"..."}` |
| `POST` | `/voice/clear` | Clear queued speech |
| `POST` | `/settings` | Update voice, camera, recording, threshold, and stream settings |
| `POST` | `/source/upload` | Upload a video for analysis |
| `POST` | `/stream/start` | Start UDP/JPEG streaming |
| `POST` | `/stream/stop` | Stop UDP/JPEG streaming |

Example health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Example voice test:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/voice/test `
  -ContentType 'application/json' `
  -Body '{"text":"Voice output is working."}'
```

## Configure an experiment

Edit `steps.json` to define laboratory steps. Each step can specify its display name, spoken text, expected objects, hold-frame count, minimum movement, and inside/outside spatial rules. The current Quick Start objects are bottle, cup, scissors, cell phone, book, and bowl.

Typical trained-model labels are `container`, `lid`, `pipette`, `sample_vial`, and `payload_rack`. Set the trained model and steps file with environment variables:

```powershell
$env:MODEL = 'best.pt'
$env:STEPS = 'steps.json'
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Other supported variables include `VIDEO`, `CAM`, `CONF`, and `IMGSZ`.

## Train a model

The `training` directory contains scripts for recording footage, importing CVAT YOLO labels, checking data, splitting datasets, training, and evaluation:

```powershell
cd training
python record_video.py
python cvat_import.py export.zip
python check_dataset.py
python split_dataset.py
python train.py
python evaluate.py
```

Training copies `best.pt` beside `main.py`; evaluation writes `model_metrics.json`. Optional newer MediaPipe hand tracking requires `hand_landmarker.task` beside `main.py`.

## Unattended voice mode

`run-unattended.bat` is intended for Windows control-room use. It starts Docker Desktop, starts the Compose service, waits for `/state`, and opens Chrome or Edge in kiosk mode with automatic audio enabled. Use `install-unattended-startup.bat` only after testing if the monitor should start with Windows.

## Repository development

```bash
git clone https://github.com/abiraj02second-glitch/bas-ai-experiment-monitor.git
cd bas-ai-experiment-monitor
```

Do not commit `best.pt`, `yolo*.pt`, `hand_landmarker.task`, runtime logs, recordings, uploads, or Python caches. These are excluded by `.gitignore`.
