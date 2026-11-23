"""
Crowd Density & Crush-Risk Early Warning (starter version)

What it does:
  1. Detects people in each video frame (YOLOv8, pretrained)
  2. Splits the frame into a grid and counts people in each cell
  3. Draws a density heatmap (red = crowded)
  4. Shows an ALERT when the total or any single cell crosses a limit

It only counts people. It does NOT identify faces.

Usage:
  python crowd_monitor.py --video crowd.mp4 --out result.mp4
  python crowd_monitor.py --video 0                      (webcam)
  python crowd_monitor.py --video crowd.mp4 --cell-limit 6 --total-limit 60
"""
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="video file path, or 0 for webcam")
    p.add_argument("--out", default="result.mp4", help="output video file")
    p.add_argument("--rows", type=int, default=4, help="grid rows")
    p.add_argument("--cols", type=int, default=6, help="grid columns")
    p.add_argument("--cell-limit", type=int, default=5, help="alert if people in one cell >= this")
    p.add_argument("--total-limit", type=int, default=50, help="alert if total people >= this")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence (lower = finds more people)")
    p.add_argument("--model", default="yolov8n.pt",
                   help="yolov8n.pt (fast) / yolov8s.pt / yolov8m.pt (more accurate, slower)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="detection image size; 960 or 1280 helps find small/far people")
    return p.parse_args()


def main():
    a = parse_args()
    model = YOLO(a.model)  # downloads automatically on first run

    src = 0 if a.video == "0" else a.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {a.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    cell_w, cell_h = w / a.cols, h / a.rows

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # class 0 = person
        res = model(frame, classes=[0], conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        grid = np.zeros((a.rows, a.cols), dtype=np.float32)

        for box in res.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            c = min(int(cx // cell_w), a.cols - 1)
            r = min(int(cy // cell_h), a.rows - 1)
            grid[r, c] += 1
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 255), -1)

        total = int(grid.sum())
        peak = int(grid.max())

        # heatmap overlay: scale so that cell-limit = fully red
        norm = np.clip(grid / max(a.cell_limit, 1), 0, 1)
        heat = cv2.resize((norm * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        mask = cv2.resize((norm > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = frame.copy()
        overlay[mask > 0] = cv2.addWeighted(frame, 0.45, heat, 0.55, 0)[mask > 0]
        frame = overlay

        # grid lines + per-cell counts
        for r in range(a.rows):
            for c in range(a.cols):
                x, y = int(c * cell_w), int(r * cell_h)
                cv2.rectangle(frame, (x, y), (int(x + cell_w), int(y + cell_h)), (255, 255, 255), 1)
                if grid[r, c] > 0:
                    cv2.putText(frame, str(int(grid[r, c])), (x + 6, y + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        danger = total >= a.total_limit or peak >= a.cell_limit
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 160) if danger else (40, 40, 40), -1)
        text = f"People: {total}   Busiest cell: {peak}"
        text += "   ALERT: CROWD RISK" if danger else "   Status: OK"
        cv2.putText(frame, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)
        cv2.imshow("Crowd Monitor (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {a.out}")


if __name__ == "__main__":
    main()
