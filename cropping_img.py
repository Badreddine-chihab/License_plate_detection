import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Initialize YOLO models
model_car = YOLO('yolov8n.pt')  # YOLOv8 model for cars
model_plate = YOLO('license_plate_model.pt')  # Custom YOLO model for license plates

# Input and output directories
input_dir = './images'
output_dir = './crop_img'
os.makedirs(output_dir, exist_ok=True)

# Class IDs for cars (COCO dataset)
car_class_ids = [2]

# Process images
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(input_dir, image_file)
    frame = cv2.imread(image_path)

    if frame is not None:
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
            license_plate_crop_blur = cv2.GaussianBlur(license_plate_crop_gray, (5,5),0)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # Save the cropped license plate image
            cropped_path = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}_plate_{i}.png')
            cv2.imwrite(cropped_path, license_plate_crop_thresh)

print('Cropping completed and images saved in crop_img.')
