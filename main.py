from fastapi import FastAPI, UploadFile, File, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import shutil
import os
from detect import detect_objects, process_video

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    output_img = detect_objects(img)
    _, img_encoded = cv2.imencode(".jpg", output_img)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")

@app.post("/detect-video/")
async def detect_video_api(file: UploadFile = File(...)):
    video_path = f"{UPLOAD_FOLDER}/{file.filename}"
    output_path = f"{OUTPUT_FOLDER}/processed_{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    process_video(video_path, output_path)

    return FileResponse(output_path, media_type="video/mp4", filename=f"processed_{file.filename}")
