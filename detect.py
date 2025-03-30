from ultralytics import YOLO
import cv2
import os

model = YOLO("models/v3/best.pt")

def detect_objects(img):
    results = model(img, conf=0.1)[0]

    # Vẽ bounding box
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item() 
        cls = int(box.cls[0])

        label = f"{model.names[cls]}: {conf:.2f}"
        color = (51, 102, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = detect_objects(frame)
        out.write(processed_frame)

    cap.release()
    out.release()