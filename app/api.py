from fastapi import FastAPI, UploadFile, File
from app.model_runner import GazeModel
import numpy as np
import cv2
import tempfile
import shutil
import os

app = FastAPI()
model = GazeModel()

@app.post("/gaze_tracking/predict_frame")
async def predict_frame(file: UploadFile = File(...)):
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = model.analyze_frame(frame)
    return result

@app.get("/gaze_tracking/health")
def health_check():
    return {"status": "running"}

@app.post("/gaze_tracking/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = model.analyze_video(tmp_path, 5)
        return result
    finally:
        os.remove(tmp_path)
