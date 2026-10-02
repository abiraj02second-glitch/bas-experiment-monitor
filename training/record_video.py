"""Record experiment videos from your webcam for labelling in CVAT.
Run from training/:  python record_video.py
Keys:  R = start/stop recording   Q = quit      Videos are saved in training/videos/"""
import cv2, os, time
CAM = 0
os.makedirs("videos", exist_ok=True)
cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(CAM)
if not cap.isOpened(): raise SystemExit("Camera not found. Try CAM = 1, or close other apps using the camera.")
writer, count = None, len(os.listdir("videos"))
print("Press R to start/stop recording, Q to quit.")
while True:
    ok, frame = cap.read()
    if not ok: break
    show = frame.copy()
    if writer:
        writer.write(frame); cv2.circle(show, (25, 25), 10, (0, 0, 255), -1)
        cv2.putText(show, "REC", (45, 32), 0, .8, (0, 0, 255), 2)
    cv2.imshow("Recorder  (R = record, Q = quit)", show)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("r"):
        if writer: writer.release(); writer = None; print("saved")
        else:
            count += 1; h, w = frame.shape[:2]
            writer = cv2.VideoWriter(f"videos/exp_{count:03d}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            print(f"recording videos/exp_{count:03d}.mp4")
    elif k == ord("q"): break
if writer: writer.release()
cap.release(); cv2.destroyAllWindows()
