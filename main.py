"""BAS AI Experiment Monitor - ONE backend: AI pipeline + API + serves the dashboard.
Run:  uvicorn main:app --host 0.0.0.0 --port 8000     then open http://localhost:8000
Env:  MODEL=best.pt  VIDEO=clip.mp4  CAM=0  CONF=0.25  IMGSZ=640"""
import math, json, time, threading, queue, os, socket, datetime as dt, tempfile
import numpy as np, cv2
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from voice_agent import VoiceSoundAgent

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
# Quick Start mode: no trained model yet -> standard YOLO model + steps built from everyday objects
_model = os.getenv("MODEL", "best.pt"); _pre = os.path.basename(_model).lower().startswith(("yolo11", "yolov8"))
QUICK = not os.getenv("STEPS") and (_pre or not os.path.exists(_model))
MODEL_PATH = _model if (_pre or not QUICK) else "yolo11n.pt"          # yolo11n.pt is downloaded once, then works offline
STEPS_FILE = os.getenv("STEPS") or ("steps_quickstart.json" if QUICK else "steps.json")
_cfg = json.load(open(STEPS_FILE))
NAME = _cfg.get("name", "Experiment") if isinstance(_cfg, dict) else "Experiment"
STEPS = _cfg["steps"] if isinstance(_cfg, dict) else _cfg
VIDEO = os.getenv("VIDEO")
print(f"MODE: {'QUICK START (standard objects, model=' + MODEL_PATH + ')' if QUICK else 'TRAINED MODEL (' + MODEL_PATH + ')'}  steps={STEPS_FILE}")
CAM, CONF, IMGSZ = int(os.getenv("CAM", 0)), float(os.getenv("CONF", 0.25)), int(os.getenv("IMGSZ", 640))
CAMS = ["Cabin Cam 1 (forward)", "Cabin Cam 2 (aft port)", "Glovebox Cam", "Laptop webcam"]
for d in ("recordings", "uploads"): os.makedirs(d, exist_ok=True)
hms = lambda: dt.datetime.now().strftime("%H:%M:%S")

# ---- Agent 1: perception (objects + hands) ----
class PerceptionAgent:
    def __init__(s):
        s.yolo = s.hands = s.mode = None
        try:
            from ultralytics import YOLO; s.yolo = YOLO(MODEL_PATH)
        except Exception as e: print("YOLO OFF:", e)
        try:
            import mediapipe as mp; s.mp = mp
            if hasattr(mp, "solutions"):
                s.hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0); s.mode = "solutions"
            else:   # newer mediapipe: needs hand_landmarker.task next to main.py
                from mediapipe.tasks import python as mpt; from mediapipe.tasks.python import vision
                s.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                    base_options=mpt.BaseOptions(model_asset_path="hand_landmarker.task"), num_hands=2)); s.mode = "tasks"
            print("Hands ON:", s.mode)
        except Exception as e: print("Hands OFF (objects only):", e)
    def run(s, frame):
        objs, hands = [], []; h, w = frame.shape[:2]
        if s.yolo:
            r = s.yolo(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
            for b in r.boxes: objs.append((r.names[int(b.cls)], *map(int, b.xyxy[0]), float(b.conf)))
        if s.hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if s.mode == "solutions": lms = [l.landmark for l in (s.hands.process(rgb).multi_hand_landmarks or [])]
            else: lms = s.hands.detect(s.mp.Image(image_format=s.mp.ImageFormat.SRGB, data=rgb)).hand_landmarks
            for lm in lms:
                xs = [p.x*w for p in lm]; ys = [p.y*h for p in lm]
                hands.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
        return objs, hands

# ---- Agent 2: step recognition + movement correctness ----
class StepRecognitionAgent:
    """Step = hand touches the right object for `hold` frames AND rules pass: min_move / inside / outside."""
    def __init__(s):
        ids = [x["id"] for x in STEPS]
        s.count = {i: 0 for i in ids}; s.last = {i: 0.0 for i in ids}
        s.trail = {i: [] for i in ids}; s.reason = {i: "" for i in ids}; s.conf = {i: 0 for i in ids}
    @staticmethod
    def overlap(a, b): return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
    @staticmethod
    def within(objs, name, cx, cy): return any(o[0] == name and o[1] <= cx <= o[3] and o[2] <= cy <= o[4] for o in objs)
    def check(s, st, objs, size):
        tr = s.trail[st["id"]]; H, W = size; cx, cy = tr[-1][0]*W, tr[-1][1]*H
        if "min_move" in st and math.dist(tr[0], tr[-1]) < st["min_move"]: return False, "the movement was too small"
        if "inside" in st and not s.within(objs, st["inside"], cx, cy): return False, st.get("zone_msg", f"not inside the {st['inside']}")
        if "outside" in st and s.within(objs, st["outside"], cx, cy): return False, st.get("zone_msg", f"still inside the {st['outside']}")
        return True, ""
    def run(s, objs, hands, size, expected):
        H, W = size
        for st in STEPS:
            i = st["id"]
            m = next((o for o in objs if o[0] in st["objects"] and (not hands or any(s.overlap(o[1:5], h) for h in hands))), None)
            if m:
                s.count[i] += 1; s.conf[i] = int(m[5]*100)
                s.trail[i] = (s.trail[i] + [((m[1]+m[3])/2/W, (m[2]+m[4])/2/H)])[-30:]
                if s.count[i] >= st["hold"] and time.time() - s.last[i] > 8:
                    ok, why = s.check(st, objs, size)
                    if ok: s.count[i] = 0; s.last[i] = time.time(); return i, True, "", s.conf[i]
                    s.reason[i] = why
            else:
                failed = s.count[i] >= st["hold"] and i == expected and time.time() - s.last[i] > 8
                why = s.reason[i]; s.count[i] = 0; s.trail[i] = []; s.reason[i] = ""
                if failed and why: s.last[i] = time.time(); return i, False, why, s.conf[i]
        return None

# ---- Agent 3: sequence validator ----
class SequenceValidatorAgent:
    def __init__(s): s.reset()
    def reset(s): s.status = {x["id"]: "pending" for x in STEPS}
    def next_step(s): return next((x for x in STEPS if s.status[x["id"]] == "pending"), None)
    def apply(s, sid):
        nx = s.next_step()
        if s.status[sid] != "pending" or nx is None: return []      # already handled: ignore
        ev = []
        for k in range(nx["id"], sid): s.status[k] = "skipped"; ev.append(("skipped", k))
        s.status[sid] = "done"; ev.append(("done", sid)); return ev

# ---- Agent 4: voice + sound management (offline TTS) ----
class AlertAgent(VoiceSoundAgent):
    """Compatibility wrapper retained for the orchestrator's existing API."""
    pass

# ---- Orchestrator (Agents 5+6: logger, stream/record) ----
class Orchestrator:
    def __init__(s):
        s.p, s.r, s.v, s.a = PerceptionAgent(), StepRecognitionAgent(), SequenceValidatorAgent(), AlertAgent()
        s.lock, s.jpeg, s.entries, s.events, s.flash = threading.Lock(), b"", [], [], None
        s.running = s.started = False; s.cam_on = True; s.source = VIDEO; s.cam_index = CAM; s.reopen = False
        s.fps, s.confidence, s.eseq, s.lseq, s.stream, s.detected = 0.0, 0, 0, 0, None, False
        s.set = {"voice": True, "recording": True, "threshold": 80, "streaming": True,
                 "resolution": None, "fps": None, "voice_volume": 1.0,
                 "voice_rate": 165, "voice_name": ""}
        s.a.configure(volume=s.set["voice_volume"], rate=s.set["voice_rate"], voice=s.set["voice_name"])
        s.reset(); threading.Thread(target=s.loop, daemon=True).start()
    # events / logs
    def event(s, level, text, voice=None):
        # Every visible backend event carries a spoken message. Explicit voice text
        # remains available for shorter, clearer spoken wording.
        spoken = voice or text
        s.eseq += 1
        s.events.insert(0, {
            "id": s.eseq,
            "level": level,
            "time": hms(),
            "text": text,
            "voice": spoken,
        }); del s.events[40:]
        if spoken and s.set["voice"]:
            priority = 10 if level in ("error", "warning") else 5
            s.a.say(spoken, priority=priority)
    def log(s, step, status, conf, detected=None, expected=None):
        st = STEPS[step-1]; s.lseq += 1
        s.entries.append({"id": s.lseq, "time": hms(), "step": step, "status": status, "conf": conf,
            "detected": detected or (st.get("done", st["name"]) if status in ("done", "low_conf") else "—"),
            "expected": expected or st.get("activity", st["name"])})
        tmp = "experiment_log.json.tmp"
        with open(tmp, "w", encoding="utf-8") as fh: json.dump(s.entries, fh, indent=1)
        os.replace(tmp, "experiment_log.json")
    def set_flash(s, **kw): s.lseq += 1; s.flash = {"id": s.lseq, "ts": time.time(), **kw}
    # control
    def reset(s):
        s.v.reset(); s.r.__init__(); s.entries.clear(); s.events.clear(); s.flash = None; s.running = s.started = False
        s.event(
            "info",
            "System ready. Press Start to begin the experiment." + (" Quick-start mode: hold the listed everyday objects up to the camera. Train your own model for the lab objects." if QUICK else ""),
            "System ready. Press Start to begin the experiment.",
        )
    def control(s, cmd):
        if cmd == "reset": return s.reset()
        if cmd == "pause": s.running = False; return s.event("info", "Experiment paused.", "Experiment paused.")
        if s.v.next_step() is None: s.reset()
        first = not s.started; s.running = s.started = True; f = STEPS[0]
        if first: s.event("info", f"Experiment started. Step 1 — {f['name']}.", f"Experiment started. Step 1. {f.get('speak', f['name'])}")
        else: s.event("info", "Experiment resumed.", "Experiment resumed.")
    def after_step(s):
        n = s.v.next_step()
        if n is None:
            s.running = False; sk = sum(1 for x in s.v.status.values() if x == "skipped")
            finished = f"Experiment finished with {sk} skipped step{'s' if sk > 1 else ''}." if sk else "Experiment complete. All steps validated."
            spoken = f"Warning. {finished}" if sk else finished
            s.event("warning" if sk else "success", finished, spoken)
        else: s.event("info", f"Next: Step {n['id']} — {n['name']}", f"Next, step {n['id']}. {n.get('speak', n['name'])}")
    def handle(s, sid, conf=0):
        for kind, k in s.v.apply(sid):
            st = STEPS[k-1]
            if kind == "done":
                s.log(k, "done", conf); s.event("success", f"Step {k} completed successfully — {st['name']}", f"Step {k} completed. {st['name']}.")
                if 0 < conf < s.set["threshold"]:
                    s.log(k, "low_conf", conf); s.event("warning", f"Low confidence detection ({conf}%)")
            else:
                msg = f"Step {k} — {st['name']} was skipped."; s.log(k, "skipped", 0)
                s.event("warning", msg, f"Warning. Step {k}, {st['name']}, was skipped.")
                s.set_flash(type="skipped", step=k, text=msg)
        s.after_step()
    def error(s, sid, detected, conf=88, voice="Warning. Incorrect experiment sequence."):
        nx = s.v.next_step() or STEPS[-1]; exp = nx.get("activity", nx["name"])
        s.log(nx["id"], "error", conf, detected, exp)
        s.event("error", f"Incorrect sequence detected. Expected \"{nx['name']}\", detected \"{detected}\".", voice)
        s.set_flash(type="error", expected=nx["name"], detected=detected, activity=detected)
    def simulate(s, what):
        nx = s.v.next_step()
        if nx is None: return
        if what == "skip":      # Skip the CURRENT expected step, including Step 6.
            k = nx["id"]; st = STEPS[k-1]; s.v.status[k] = "skipped"; msg = f"Step {k} — {st['name']} was skipped."
            s.log(k, "skipped", 0); s.event("warning", msg, f"Warning. Step {k}, {st['name']}, was skipped.")
            s.set_flash(type="skipped", step=k, text=msg)
            return s.after_step()
        if what == "wrong":
            others = [x for x in STEPS if x["id"] != nx["id"] and s.v.status[x["id"]] == "pending"] or STEPS
            return s.error(nx["id"], others[-1].get("activity", others[-1]["name"]))
        s.handle(int(what), 95)
    # settings / source / stream
    def apply_setting(s, k, v):
        if k in ("voice", "recording", "streaming"): s.set[k] = bool(v)
        elif k == "threshold": s.set[k] = int(v)
        elif k == "resolution": s.set[k] = v; s.reopen = True
        elif k == "fps": s.set[k] = int(v); s.reopen = True
        elif k == "voice_volume": s.set[k] = max(0.0, min(1.0, float(v)))
        elif k == "voice_rate": s.set[k] = max(80, min(300, int(v)))
        elif k == "voice_name": s.set[k] = str(v or "")
        elif k == "camera" and v in CAMS: s.cam_index = CAMS.index(v); s.source = None; s.cam_on = True; s.reopen = True
        if k == "streaming" and not v: s.stream = None
        s.a.configure(enabled=s.set["voice"], volume=s.set["voice_volume"],
                      rate=s.set["voice_rate"], voice=s.set["voice_name"])
    def open_cap(s):
        if s.source: return cv2.VideoCapture(s.source)
        cap = cv2.VideoCapture(s.cam_index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(s.cam_index)
        if s.set["resolution"]:
            w, h = map(int, s.set["resolution"].split("x")); cap.set(3, w); cap.set(4, h)
        if s.set["fps"]: cap.set(cv2.CAP_PROP_FPS, s.set["fps"])
        return cap
    def placeholder(s, text="CAMERA OFF"):
        img = np.zeros((360, 640, 3), np.uint8); cv2.putText(img, text, (150 if len(text) > 10 else 190, 190), 0, 1.3, (120, 120, 120), 3)
        return cv2.imencode(".jpg", img)[1].tobytes()
    def push(s, f):
        if s.stream and s.set["streaming"]:
            ip, port, sock = s.stream; ok, j = cv2.imencode(".jpg", cv2.resize(f, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 50])
            if ok and len(j) < 60000:
                try: sock.sendto(j.tobytes(), (ip, port))
                except Exception: pass
    # main loop
    def loop(s):
        cap = writer = None; src_fps = 20; last = time.time(); n = 0
        while True:
            if not s.cam_on and not s.source:
                if cap: cap.release(); cap = None
                if writer: writer.release(); writer = None
                with s.lock: s.jpeg = s.placeholder()
                s.fps = 0; time.sleep(0.2); continue
            if cap is None or s.reopen:
                if cap: cap.release()
                if writer: writer.release(); writer = None
                cap = s.open_cap(); s.reopen = False; src_fps = cap.get(cv2.CAP_PROP_FPS) or 20
                if not cap.isOpened(): s.event("error", "Camera could not be opened. Check the camera index (CAM=), permissions, or that no other app is using it.")
            ok, f = cap.read()
            if not ok:
                if s.source: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                elif not cap.isOpened():
                    with s.lock: s.jpeg = s.placeholder("NO CAMERA SIGNAL")
                time.sleep(0.1); continue
            if s.source: time.sleep(1 / src_fps)
            objs, hands = s.p.run(f)
            s.detected = bool(objs or hands)
            if objs: s.confidence = int(np.mean([o[5] for o in objs]) * 100)
            if s.running:
                nx = s.v.next_step(); res = s.r.run(objs, hands, f.shape[:2], nx["id"] if nx else 0)
                if res:
                    sid, okk, why, conf = res
                    if okk: s.handle(sid, conf)
                    else: s.error(sid, f"Incorrect movement: {why}", conf, f"Warning. Step {sid} not done correctly. {why}.")
            for o in objs:
                cv2.rectangle(f, (o[1], o[2]), (o[3], o[4]), (0, 255, 0), 2); cv2.putText(f, f"{o[0]} {int(o[5]*100)}%", (o[1], o[2]-5), 0, .55, (0, 255, 0), 2)
            for h in hands: cv2.rectangle(f, h[:2], h[2:], (255, 120, 0), 2)
            if s.set["recording"]:
                if writer is None:
                    hh, ww = f.shape[:2]; writer = cv2.VideoWriter(f"recordings/{int(time.time())}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (ww, hh))
                writer.write(f)
            elif writer: writer.release(); writer = None
            n += 1
            if n % 2 == 0: s.push(f)
            with s.lock: s.jpeg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            now = time.time(); s.fps = 0.9 * s.fps + 0.1 / max(now - last, 1e-3); last = now
    def state(s):
        nx = s.v.next_step(); fl = s.flash if s.flash and time.time() - s.flash["ts"] < 8 else None
        st = s.stream
        return {"name": NAME, "quick": QUICK, "running": s.running, "started": s.started, "camera": bool(s.cam_on or s.source),
            "source": "file" if s.source else "camera", "detected": s.detected, "fps": int(s.fps), "confidence": s.confidence,
            "progress": int(min(100, 100 * s.r.count[nx["id"]] / nx["hold"])) if nx else 0,
            "next": {"id": nx["id"]} if nx else None,
            "steps": [{"id": x["id"], "name": x["name"], "activity": x.get("activity", x["name"]), "done": x.get("done", x["name"]),
                       "status": s.v.status[x["id"]]} for x in STEPS],
            "events": s.events, "log": s.entries[-200:], "flash": fl,
            "voice": s.a.snapshot(),
            "stream": {"active": bool(st), "ip": st[0] if st else "", "port": st[1] if st else 0}, "settings": s.set}

O = Orchestrator()
app = FastAPI(); app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def index(): return FileResponse("static/index.html")
@app.get("/manus-routes.json")
def routes_manifest(): return FileResponse("manus-routes.json", media_type="application/json")
@app.get("/logo.svg")
def logo(): return FileResponse("static/logo.svg", media_type="image/svg+xml")
@app.get("/state")
def state(): return O.state()
@app.get("/health")
def health():
    """Operational health endpoint for Docker, kiosk startup, and monitoring."""
    return {"ok": True, "service": "bas-monitor", "camera": O.state()["camera"],
            "running": O.running, "voice": O.a.snapshot(), "model": MODEL_PATH}
@app.get("/events")
def events(after: int = 0):
    """Return only events newer than a sequence id for reliable integrations."""
    return {"events": [e for e in O.events if e["id"] > max(0, int(after))], "latest": O.eseq}
@app.get("/video")     # MJPEG: open http://<edge-ip>:8000/video on any machine
def video():
    def gen():
        while True:
            with O.lock: j = O.jpeg
            if j: yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            time.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")
@app.post("/control/{cmd}")
def control(cmd: str):
    if cmd not in {"start", "pause", "reset"}: raise HTTPException(400, "Unsupported control command")
    O.control(cmd); return {"ok": True, "state": O.state()}
@app.post("/simulate/{what}")
def simulate(what: str): O.simulate(what); return {"ok": True}
@app.post("/settings")
def settings(body: dict):
    key, value = body.get("key"), body.get("value")
    allowed = {"voice", "recording", "streaming", "threshold", "resolution", "fps",
               "voice_volume", "voice_rate", "voice_name", "camera"}
    if key not in allowed: raise HTTPException(400, f"Unsupported setting: {key}")
    if key in {"voice", "recording", "streaming"}:
        if isinstance(value, str): value = value.lower() in {"1", "true", "yes", "on"}
        else: value = bool(value)
    try: O.apply_setting(key, value)
    except (TypeError, ValueError) as exc: raise HTTPException(400, str(exc))
    return {"ok": True, "settings": O.set, "voice": O.a.snapshot()}
@app.post("/voice/test")
def voice_test(body: dict | None = None):
    body = body or {}
    text = str(body.get("text") or "Voice output is working. Notifications are enabled.")
    if len(text) > 300: raise HTTPException(400, "Voice test text is limited to 300 characters")
    queued = O.a.say(text, priority=20, dedupe_seconds=0)
    O.event("info", "Voice output test queued.", text if queued else "Voice output test could not be queued.")
    return {"ok": queued, "voice": O.a.snapshot()}
@app.post("/voice/clear")
def voice_clear():
    O.a.clear(); return {"ok": True, "voice": O.a.snapshot()}
@app.post("/camera/{mode}")
def camera(mode: str):
    O.cam_on = mode == "on"; O.source = None; O.reopen = True
    message = "Camera started." if O.cam_on else "Camera stopped."
    O.event("info", message, message)
    return {"camera": O.cam_on}
@app.post("/source/upload")
async def upload(request: Request, name: str = "clip.mp4"):
    if request.headers.get("content-length") and int(request.headers["content-length"]) > 2_000_000_000:
        raise HTTPException(413, "Upload is limited to 2 GB")
    path = os.path.join("uploads", os.path.basename(name))
    with open(path, "wb") as fh:
        async for chunk in request.stream(): fh.write(chunk)
    O.source = path; O.reopen = True
    O.event("info", f"Analysing uploaded video: {name}", "Video uploaded. Analysis started.")
    return {"ok": True}
@app.post("/stream/start")
def stream_start(body: dict):
    ip = str(body.get("ip", "")).strip()
    try: port = int(body.get("port"))
    except (TypeError, ValueError): raise HTTPException(400, "A numeric UDP port is required")
    if not ip or not (1 <= port <= 65535): raise HTTPException(400, "A valid destination IP and port are required")
    try: socket.getaddrinfo(ip, port, type=socket.SOCK_DGRAM)
    except OSError as exc: raise HTTPException(400, f"Invalid stream destination: {exc}")
    if O.stream:
        try: O.stream[2].close()
        except Exception: pass
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); O.stream = (ip, port, sock)
    O.event("success", f"Streaming video to {ip}:{port} (UDP/JPEG).", "Video streaming started."); return {"ok": True, "stream": {"ip": ip, "port": port}}
@app.post("/stream/stop")
def stream_stop():
    if O.stream:
        try: O.stream[2].close()
        except Exception: pass
    O.stream = None; O.event("info", "Video streaming stopped.", "Video streaming stopped."); return {"ok": True}
@app.get("/model")
def model():
    try: return json.load(open("model_metrics.json"))
    except Exception: return {"name": "YOLO11 + MediaPipe Hands", "version": "not evaluated yet", "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "processing": "EDGE / LOCAL"}
@app.get("/log.json")
def log_json(): return FileResponse("experiment_log.json") if os.path.exists("experiment_log.json") else []
