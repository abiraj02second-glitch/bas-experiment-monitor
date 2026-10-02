"""Check CVAT labels after cvat_import.py. Run from training/:  python check_dataset.py
Prints images/labels per class, flags problems, and saves preview images with boxes to dataset/preview/."""
import glob, os, re, cv2, collections
names = {int(k): v.strip() for k, v in re.findall(r"^\s+(\d+):\s*(\S+)", open("data.yaml").read(), re.M)}
imgs = sorted(glob.glob("dataset/images/all/*.*"))
count, empty, bad, os_ = collections.Counter(), [], [], os.makedirs("dataset/preview", exist_ok=True)
for n, img in enumerate(imgs):
    base = os.path.splitext(os.path.basename(img))[0]
    lb = f"dataset/labels/all/{base}.txt"
    rows = [l.split() for l in open(lb).read().splitlines() if l.strip()] if os.path.exists(lb) else []
    if not rows: empty.append(base); continue
    frame = cv2.imread(img); h, w = frame.shape[:2]
    for r in rows:
        c, x, y, bw, bh = int(r[0]), *map(float, r[1:5])
        if c not in names or not (0 <= x <= 1 and 0 <= y <= 1): bad.append((base, r)); continue
        count[names[c]] += 1
        p1, p2 = (int((x-bw/2)*w), int((y-bh/2)*h)), (int((x+bw/2)*w), int((y+bh/2)*h))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2); cv2.putText(frame, names[c], p1, 0, .6, (0, 255, 0), 2)
    if n % 10 == 0: cv2.imwrite(f"dataset/preview/{base}.jpg", frame)   # every 10th image
print(f"\nImages: {len(imgs)}   without labels: {len(empty)}   bad label rows: {len(bad)}")
print("\nBoxes per class (aim for 150+ each):")
for i, nm in names.items(): print(f"  {nm:12} {count[nm]:5}  {'OK' if count[nm] >= 150 else 'LOW' if count[nm] else 'MISSING'}")
print("\nOpen dataset/preview/ and confirm the boxes sit on the right objects.")
