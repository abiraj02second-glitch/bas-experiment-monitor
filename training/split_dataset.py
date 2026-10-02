"""Step 3 (after labelling in Roboflow/CVAT/labelImg, YOLO format): 80/20 train/val split.
Expects dataset/images/all/*.jpg and dataset/labels/all/*.txt"""
import glob, os, random, shutil
imgs = glob.glob("dataset/images/all/*.jpg"); random.seed(0); random.shuffle(imgs)
cut = int(len(imgs) * 0.8)
for name, part in (("train", imgs[:cut]), ("val", imgs[cut:])):
    for sub in ("images", "labels"): os.makedirs(f"dataset/{sub}/{name}", exist_ok=True)
    for p in part:
        shutil.copy(p, f"dataset/images/{name}")
        lb = p.replace("images", "labels")[:-4] + ".txt"
        if os.path.exists(lb): shutil.copy(lb, f"dataset/labels/{name}")
print(len(imgs[:cut]), "train /", len(imgs[cut:]), "val")
