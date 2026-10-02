"""Test the trained model on the validation (20%) images. Run from training/:  python evaluate.py --imgsz 640
Prints precision / recall / mAP per class, explains weak classes, and recommends a confidence threshold."""
import os, argparse, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "..", "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    from ultralytics import YOLO
    data = os.path.join(HERE, "data_abs.yaml")
    if not os.path.exists(data): raise SystemExit("Run train.py once first (it creates data_abs.yaml).")
    m = YOLO(a.model)
    r = m.val(data=data, imgsz=a.imgsz, split="val", plots=True, project=os.path.join(HERE, "runs"), name="eval", exist_ok=True)
    b = r.box
    print(f"\n{'class':14}{'precision':>10}{'recall':>9}{'mAP50':>8}{'mAP50-95':>10}   verdict")
    seen = set()
    for i, ci in enumerate(b.ap_class_index):
        seen.add(int(ci)); P, R, A50, A = b.p[i], b.r[i], b.ap50[i], b.ap[i]
        if A50 >= 0.8: v = "good"
        elif R < 0.6: v = "MISSES objects -> add more/varied labelled images, or use --imgsz 800 / a bigger model"
        elif P < 0.6: v = "FALSE alarms -> label more frames where the object is absent/similar, or raise CONF"
        else: v = "weak -> more data"
        print(f"{r.names[int(ci)]:14}{P:10.2f}{R:9.2f}{A50:8.2f}{A:10.2f}   {v}")
    for ci, nm in r.names.items():
        if ci not in seen: print(f"{nm:14}  NOT in the validation set (or never detected) -> label more of this class")
    print(f"\nOverall mAP50 = {b.map50:.3f}   mAP50-95 = {b.map:.3f}   (target: mAP50 above 0.8)")
    import json   # feeds the dashboard's "Model" page with your real numbers
    json.dump({"name": "YOLO11 + MediaPipe Hands", "version": "trained", "accuracy": round(float(b.map50)*100, 1),
               "precision": round(float(np.mean(b.p))*100, 1), "recall": round(float(np.mean(b.r))*100, 1),
               "f1": round(float(np.mean(b.f1))*100, 1), "processing": "EDGE / LOCAL"}, open(os.path.join(HERE, "..", "model_metrics.json"), "w"))
    try:
        f1 = np.array(b.f1_curve).mean(0); conf = float(np.array(b.px)[f1.argmax()])
        print(f"Recommended confidence threshold: {conf:.2f}   ->  PowerShell: $env:CONF=\"{conf:.2f}\"")
    except Exception: pass
    vdir = os.path.join(DATASET := os.path.join(HERE, "dataset"), "images", "val")
    m.predict(source=vdir, conf=0.25, imgsz=a.imgsz, save=True, project=os.path.join(HERE, "runs"), name="val_preds", exist_ok=True, verbose=False)
    print("Look at the predicted boxes on validation images in:", os.path.join(HERE, "runs", "val_preds"))
    print("Plots (confusion matrix, PR curve):", os.path.join(HERE, "runs", "eval"))

if __name__ == "__main__":
    main()
