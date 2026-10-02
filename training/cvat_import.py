"""Import a CVAT export (format: YOLO 1.1, 'Save images' ticked) into the project dataset.
Usage:  python cvat_import.py path\\to\\export.zip
Run from the training/ folder. Then run split_dataset.py and train.py."""
import sys, zipfile, os, shutil, tempfile, glob
zpath = sys.argv[1]
tmp = tempfile.mkdtemp(); zipfile.ZipFile(zpath).extractall(tmp)
os.makedirs("dataset/images/all", exist_ok=True); os.makedirs("dataset/labels/all", exist_ok=True)
prefix = os.path.splitext(os.path.basename(zpath))[0] + "_"    # avoids name clashes between exports
n = 0
for txt in glob.glob(f"{tmp}/**/obj_train_data/*.txt", recursive=True):
    base = os.path.splitext(txt)[0]
    img = next((base + e for e in (".jpg", ".png", ".jpeg") if os.path.exists(base + e)), None)
    if not img: continue
    name = prefix + os.path.basename(base)
    shutil.copy(img, f"dataset/images/all/{name}{os.path.splitext(img)[1]}")
    shutil.copy(txt, f"dataset/labels/all/{name}.txt"); n += 1
print("imported", n, "labelled frames")
names = glob.glob(f"{tmp}/**/obj.names", recursive=True)
if names:
    print("\nCVAT class order (must match data.yaml and steps.json):")
    for i, l in enumerate(open(names[0]).read().split()): print(f"  {i}: {l}")
