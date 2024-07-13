from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import os
from PIL import Image
import io
import json
import cv2
from segmentation import get_yolov5, get_image_from_bytes
from starlette.responses import Response
from tqdm import tqdm  # Import tqdm for progress bar
import random
from fastapi.middleware.cors import CORSMiddleware 

model = get_yolov5()

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
    description="""Obtain object value out of image or video and return image and JSON result""",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = [
    "bike", "bus", "car", "motor", "person", "rider", 
    "traffic light", "traffic sign", "train", "truck", 
    "direct area", "dashed lane", "alter area", "solid lane"
]

# Define distinct bright colors for each class label
class_colors = [
    (255, 0, 0),      # bike (Red)
    (0, 255, 0),      # bus (Green)
    (0, 0, 255),      # car (Blue)
    (255, 255, 0),    # motor (Yellow)
    (255, 0, 255),    # person (Purple)
    (0, 255, 255),    # rider (Cyan)
    (255, 128, 0),    # traffic light (Orange)
    (128, 0, 255),    # traffic sign (Magenta)
    (0, 128, 255),    # train (Light Blue)
    (128, 255, 0),    # truck (Lime)
    (255, 128, 128),  # direct area (Light Red)
    (128, 255, 128),  # dashed lane (Light Green)
    (128, 128, 255),  # alter area (Light Blue)
    (255, 255, 128)   # solid lane (Light Yellow)
]

@app.post("/object-to-video", response_model=dict)
async def object_detection(file: UploadFile = File(...)):
    # Check file extensions
    allowed_image_extensions = {".jpg", ".jpeg", ".png", ".gif"}
    allowed_video_extensions = {".mp4", ".avi", ".mov"}
    file_extension = os.path.splitext(file.filename)[1]

    if file_extension.lower() in allowed_image_extensions:
        # Handle image processing
        input_image = Image.open(io.BytesIO(await file.read()))
        results = model(input_image)
        detect_res = results.pandas().xyxy[0].to_json(orient="records")
        detect_res = json.loads(detect_res)
        return {"result": detect_res}

    elif file_extension.lower() in allowed_video_extensions:
        # Handle video processing
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as video_file:
            shutil.copyfileobj(file.file, video_file)

        cap = cv2.VideoCapture(temp_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = "./output_video.mp4"
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc="Processing frames") as pbar:  # Initialize tqdm progress bar
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = model(input_image)
                detect_res = results.pandas().xyxy[0].to_json(orient="records")
                detect_res = json.loads(detect_res)

                for detection in detect_res:
                    xmin, ymin, xmax, ymax, label = [int(detection["xmin"]), int(detection["ymin"]), int(detection["xmax"]), int(detection["ymax"]), detection["name"]]
                    # Use the color assigned to the class
                    color = class_colors[class_names.index(label)]
                    thickness = 2
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
                    frame = cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                out.write(frame)
                pbar.update(1)  # Update progress bar

        cap.release()
        out.release()
        os.remove(temp_video_path)

        print("video processed")
        # Return the video file as a binary response
        return FileResponse(output_path)

    else:
        return {"error": "Unsupported file format"}
@app.post("/object-to-img")
async def detect_img_return_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.ims:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")