# 👥 Crowd Safety AI

Crowd density estimation and crush-risk early warning from video.

Upload a crowd video and the system counts people, splits the scene into a grid,
draws a density heatmap, classifies every moment as **Safe / Watch / Danger**, and
produces an alert report. It counts people only. **No faces are identified.**

> Built as a learning project after the 2025 Karur rally crowd crush, to explore how
> computer vision could help spot dangerous crowding earlier.

![dashboard screenshot](docs/dashboard.png)
<!-- Take a screenshot of your dashboard, save it as docs/dashboard.png -->

## Features

- Person detection with a pretrained **YOLOv8** model (choose n / s / m size)
- Grid-based **density heatmap** drawn on the video
- **Risk levels**: Safe, Watch (60% of a limit), Danger (limit reached)
- **Live dashboard**: frame, status sign, counts, crowd-size chart, zone heat and an alert
  log update while the footage is analysed
- **Three sources**: video file, webcam, or a camera stream (RTSP / HTTP, for example a phone
  running an IP-camera app on the same Wi-Fi). Webcam and streams need the app running on
  your own computer, not on a hosted Space
- Interactive Streamlit app: scrub through the footage, narrow the time window, and
  change the alert limits with results updating instantly (no re-analysis)
- Status sign that turns Safe / Watch / Danger as you scrub, crowd-size chart with a
  risk strip, zone heatmap (average, peak or a single moment), alert events table
- Download the annotated video and a CSV of per-frame data
- Command-line version (`crowd_monitor.py`) for webcam or quick tests

## Tech stack

Python, OpenCV, Ultralytics YOLOv8, Streamlit, Pandas, Altair

## How it works

1. Each frame goes through YOLOv8 (class `person`).
2. The centre of every box is assigned to a cell of an N x M grid.
3. Risk score = max(total people / total limit, busiest cell / cell limit).
4. Score below 0.6 is **Safe**, 0.6 to 1.0 is **Watch**, 1.0 or more is **Danger**.
5. The dashboard turns the per-frame results into charts and alert events.

## Run it locally

```
pip install -r requirements.txt
python -m streamlit run app.py
```

On Windows, if `pip` is blocked by policy, use `python -m pip install -r requirements.txt`.

Command-line version (needs the full OpenCV build for the preview window):

```
pip install ultralytics opencv-python numpy
python crowd_monitor.py --video crowd.mp4 --out result.mp4 --model yolov8s.pt --imgsz 960
```

## Deploy on Hugging Face Spaces (free)

1. Create a new Space, SDK: **Streamlit**, hardware: CPU basic, Public.
2. Upload `app.py`, `requirements.txt` and `packages.txt`.
3. Wait for the build, then open the Space link.

Tip: on free CPU use `yolov8n.pt` and a short video (10 to 20 seconds).

## Tuning tips

| Problem | Try |
| --- | --- |
| Count too low | Bigger model (`yolov8s` / `yolov8m`), larger detection size (960 / 1280), lower confidence |
| False detections | Raise confidence to 0.3 to 0.4 |
| Too slow | `yolov8n`, detect every 5 frames, smaller video |
| Alerts too early or late | Change the cell limit and total limit in the sidebar |

## Limitations (please read)

- YOLO detects individuals, so in **very dense** crowds it under-counts because people
  overlap. Dedicated crowd-counting models (CSRNet, DM-Count) work better there.
- Counts are in image space. Real safety needs people per square metre, which needs
  camera calibration or a known ground area.
- Camera angle matters. High-angle shots work best.
- This is a learning prototype, **not a certified safety system**.

## Roadmap

- [ ] Crowd-counting model (CSRNet) trained on ShanghaiTech / UCF-QNRF
- [ ] Movement direction and speed to detect crowd surges
- [ ] People per square metre with ground-plane calibration
- [ ] Live camera stream and SMS / WhatsApp alerts

## Ethics and privacy

Use only videos you have the right to use. Do not try to identify individuals.
Public Spaces are open to everyone, so never upload private footage there.

## Author

**S Muthukrishnan**, AI/ML
[GitHub](https://github.com/Muthu4593) ·
[LinkedIn](https://www.linkedin.com/in/muthukrishnan-s-9959b6311)
