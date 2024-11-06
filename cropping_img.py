import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Initialize YOLO models
model_car = YOLO('yolov8n.pt')  # YOLOv8 model for cars
model_plate = YOLO('license_plate_model.pt')  # Custom YOLO model for license plates

# Input and output directories
input_dir = './Inputs'
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


            # Resize the aligned image to 3x its original size
            license_plate_resized = cv2.resize(license_plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)


            # Save the preprocessed license plate image
            cropped_path = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}_plate_{i}.png')
            cv2.imwrite(cropped_path, license_plate_resized)

print('Cropping, aligning, resizing, and preprocessing completed; images saved in crop_img.')
