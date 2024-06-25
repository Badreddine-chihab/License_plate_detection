import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Initialize YOLO models
model_car = YOLO('yolov8n.pt')  # YOLOv8 model for cars
model_plate = YOLO('license_plate_model.pt')  # Custom YOLO model for license plates

# Load video
video_path = './videos/test1.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Output directory for cropped images
output_dir = './crop_img'
os.makedirs(output_dir, exist_ok=True)

# Class IDs for cars (COCO dataset)
car_class_ids = [2]

# Process first 10 frames
frame_nmr = -1
ret = True
while ret and frame_nmr < 10:
    frame_nmr += 1
    ret, frame = cap.read()

    if ret:
        # Run YOLO inference for car detection
        detections = model_car(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            # Filter detections to show only cars
            if int(class_id) in car_class_ids:
                detections_.append([x1, y1, x2, y2, score])

        # Detect license plates
        license_plates = model_plate(frame)[0]
        for i, license_plate in enumerate(license_plates.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = license_plate

            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Save the cropped license plate image
            cropped_path = os.path.join(output_dir, f'frame_{frame_nmr:04d}_plate_{i}.png')
            cv2.imwrite(cropped_path, license_plate_crop)

# Release video capture object
cap.release()

print('Cropping completed and images saved in crop_img.')
