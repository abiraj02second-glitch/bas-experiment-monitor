"""Improved training. Run from training/:  python train.py            (balanced: yolo11s, 640px)
   Faster / lighter:  python train.py --model yolo11n.pt --imgsz 480 --epochs 60
   Best accuracy:     python train.py --model yolo11m.pt --imgsz 640 --batch 4"""
import os, sys, shutil, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset")

def check_dataset():
    problems = []
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = os.path.join(DATASET, *sub.split("/")); n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"  {sub:14} {n} files")
        if n == 0: problems.append(sub)
    if problems:
        print("Dataset not ready. Run cvat_import.py, check_dataset.py, split_dataset.py first."); sys.exit(1)

def make_abs_yaml():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(HERE, "data.yaml"))); cfg["path"] = DATASET
    out = os.path.join(HERE, "data_abs.yaml"); yaml.safe_dump(cfg, open(out, "w")); return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt"); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    check_dataset()
    from ultralytics import YOLO
    data = make_abs_yaml()
    m = YOLO(a.model)
    m.train(data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=30, workers=2, cache=True,
            # Augmentation: small rotations only. Big rotations (180) stretch the boxes and HURT accuracy.
            degrees=15, translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5,
            hsv_h=0.02, hsv_s=0.7, hsv_v=0.4, mosaic=1.0, mixup=0.1, close_mosaic=15, cos_lr=True,
            project=os.path.join(HERE, "runs"), name="bas")
    best_path = m.trainer.best
    shutil.copy(best_path, os.path.join(HERE, "..", "best.pt"))
    YOLO(best_path).export(format="onnx", imgsz=a.imgsz, simplify=True)
    print("\nDone. Copied best.pt next to main.py. Now run:  python evaluate.py --imgsz", a.imgsz)

if __name__ == "__main__":
    main()
