"""Step 1: turn recorded experiment videos into images to label. Usage: python extract_frames.py videos/ dataset/images/all 5"""
import cv2, sys, os, glob
src, out, every = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5
os.makedirs(out, exist_ok=True); n = 0
for v in glob.glob(f"{src}/*.mp4"):
    cap, i = cv2.VideoCapture(v), 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % (every * 5) == 0:   # ~1 frame per `every` * 5 source frames
            cv2.imwrite(f"{out}/{os.path.basename(v)[:-4]}_{i}.jpg", f); n += 1
        i += 1
print("saved", n, "frames")
