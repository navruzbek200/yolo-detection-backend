import asyncio
import json
import os
import threading
import time

import cv2
import yt_dlp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
YOUTUBE_URL = os.environ.get("YOUTUBE_URL", "https://www.youtube.com/live/86YA-i9Kaak")

model = YOLO(MODEL_PATH)

detections_lock = threading.Lock()
latest_detections: list = []
status = {"running": False, "error": None, "frames": 0}


def get_stream_url(url: str) -> str:
    ydl_opts = {"format": "best[height<=480]/best", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]


def inference_loop():
    while True:
        try:
            print("Fetching stream URL...")
            stream_url = get_stream_url(YOUTUBE_URL)
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                raise RuntimeError("Cannot open stream")
            status["running"] = True
            status["error"] = None
            tick = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Stream dropped, reconnecting...")
                    break
                tick += 1
                if tick % 6 != 0:   # ~5 fps inference
                    continue
                frame = cv2.resize(frame, (640, 360))
                h, w = frame.shape[:2]
                results = model(frame, conf=0.4, classes=[0], verbose=False)
                dets = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        dets.append({
                            "x": round(x1 / w * 100, 1),
                            "y": round(y1 / h * 100, 1),
                            "w": round((x2 - x1) / w * 100, 1),
                            "h": round((y2 - y1) / h * 100, 1),
                            "conf": round(float(box.conf[0]), 3),
                        })
                with detections_lock:
                    latest_detections[:] = dets
                status["frames"] += 1
            cap.release()
        except Exception as e:
            status["running"] = False
            status["error"] = str(e)
            print(f"Error: {e}, retrying in 10s...")
            time.sleep(10)


threading.Thread(target=inference_loop, daemon=True).start()


@app.get("/health")
def health():
    with detections_lock:
        n = len(latest_detections)
    return {"status": "ok", "detections": n, **status}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with detections_lock:
                dets = list(latest_detections)
            await websocket.send_text(json.dumps(dets))
            await asyncio.sleep(0.2)
    except (WebSocketDisconnect, Exception):
        pass
