# BAS AI Experiment Monitor — single package
One backend (FastAPI + AI agents) serves your dashboard. One command, one address.

## Run
Windows: double-click **run.bat** (first run creates the venv and installs; then opens http://localhost:8000)
Manual:  `venv\Scripts\activate` -> `uvicorn main:app --host 0.0.0.0 --port 8000` -> open http://localhost:8000
Other PCs on the network: http://<this-PC-IP>:8000   (raw video: http://<this-PC-IP>:8000/video)

## Automatic voice alerts

- Keep **Voice alert ON** in the dashboard.
- The first live notification is now spoken automatically, followed by every
  later notification. You do not need to press the Voice button.
- The browser automatically speaks start, next-step, skipped-step, warning,
  error, pause, reset, camera, upload, streaming and completion messages.
- **Skip Step** now skips the current expected step. This works for every
  step, including Step 3 and the final Step 6.
- If several events arrive together, the browser reads them in order instead
  of dropping the skipped-step warning.

### Fully unattended voice mode

Normal browsers may block sound until a person clicks once. For a hands-free
control-room or spacecraft display, double-click **run-unattended.bat**. It:

1. starts Docker Desktop and waits for its engine;
2. starts this BAS container and waits for the API; and
3. opens Chrome or Edge in kiosk mode with automatic audio enabled.

After testing it, double-click **install-unattended-startup.bat** once if the
monitor must start automatically whenever Windows starts. No dashboard button
is then required for the first or later voice alerts.

## Run with Docker Desktop

Open PowerShell in this `final` folder and run:

```powershell
docker compose up --build -d
docker compose ps
```

Open the complete website at **http://localhost:8000**. The frontend is already
included in `static/index.html`, so do not run `npm run dev` for this package.

View logs:

```powershell
docker compose logs -f bas-monitor
```

Stop the application:

```powershell
docker compose down
```

Docker Desktop runs the site on this computer and local network; it does not
create a public Internet URL. Also note that Docker Desktop's Linux container
may not have direct access to the Windows webcam. The dashboard and upload mode
still work, but for direct Windows webcam access run Uvicorn outside Docker.

## What is live in the dashboard
Start / Pause / Reset, step tracking, next-step prompts, skipped + wrong-step alerts (spoken by the backend, offline),
timestamped log (Logs page + CSV export, and experiment_log.json on disk), live video with detection boxes,
Start/Stop Camera, Upload Video (analysed in real time), settings (voice, recording, threshold, camera, resolution, fps),
Stream page (pushes UDP/JPEG to any IP:port; view it with `python receiver.py 5000` on that PC), Model page (real metrics).
If the backend is not running the dashboard falls back to its built-in demo.

## Quick Start mode (no training needed)
If there is no trained model (best.pt), the backend switches to Quick Start automatically: it uses the standard YOLO model
(downloaded once on first run, works offline after that) and steps_quickstart.json — everyday objects:
bottle, cup, scissors, cell phone, book, bowl. Hold each object up to the camera in order (move it a little) and watch the
dashboard track the steps, raise skipped / wrong-step alerts and speak. When best.pt exists, your lab steps.json is used instead.
Force a mode with  STEPS=steps.json  or  STEPS=steps_quickstart.json.

## Your experiment
Edit steps.json: names, spoken text, and the rules per step —
 objects   what the hand must touch          hold      frames it must be held
 min_move  minimum travel (fraction of frame) inside/outside  object must be inside / outside another detected object
Labels used: container, lid, pipette, sample_vial, payload_rack.

## Train the model  (training/ folder, in the venv)
1 python record_video.py   2 label in CVAT (YOLO 1.1, save images)   3 python cvat_import.py export.zip
4 python check_dataset.py  5 python split_dataset.py                 6 python train.py   7 python evaluate.py
train.py copies best.pt next to main.py; evaluate.py writes model_metrics.json for the Model page.
Optional hand tracking (newer mediapipe): download hand_landmarker.task into this folder:
 https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
Env options: MODEL=best.pt  VIDEO=clip.mp4  CAM=0  CONF=0.25  IMGSZ=640

## Backend voice and operations API

The backend now includes a dedicated `VoiceSoundAgent` worker. It keeps speech synthesis
off the camera/API threads, uses a bounded priority queue so urgent warnings win, removes
duplicate announcements, and exposes runtime health and queue metrics in `/state.voice`.
Every event in `/state.events` includes a `voice` message, so browser speech and native
offline TTS use the same notification source.

- `GET /health` — Docker/kiosk readiness and voice backend status.
- `GET /events?after=<id>` — incremental event feed for external displays or agents.
- `POST /voice/test` with optional `{"text":"..."}` — verify speech output.
- `POST /voice/clear` — clear queued speech.
- `POST /settings` — supports `voice`, `voice_volume` (0–1), `voice_rate` (80–300),
  and `voice_name`, in addition to the existing camera/recording settings.

The dashboard includes a live Voice agent indicator. If the browser blocks speech before
the first user interaction, use kiosk mode (`run-unattended.bat`) or press Start once;
the backend still logs and exposes every spoken notification.

## Managed permanent website

This project is configured for Manus-managed container hosting. The application listens on
the managed `PORT` (default `3000`), serves the dashboard and API from one FastAPI process,
and exposes `/health` for readiness. The published dashboard uses the same root-relative API
paths as local mode, so it does not depend on `localhost` or a Windows-specific address.

For a local Windows run, open PowerShell in the extracted project folder first, not in
`C:\Windows\System32`:

```powershell
cd C:\path\to\bas_project
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
