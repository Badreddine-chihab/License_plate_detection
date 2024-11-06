import os
import cv2
import numpy as np
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

# To store previously detected plate bounding boxes
previous_plate_boxes = []

# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate the areas of both boxes and their union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    
    # Compute the IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# IoU threshold to determine if plates are the same
iou_threshold = 0.5

# Process each frame
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    if ret:
        # Run YOLO inference for car detection
        car_detections = model_car(frame)[0]
        
        for detection in car_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            # Filter detections to show only cars
            if int(class_id) in car_class_ids:
                car_box = [x1, y1, x2, y2]

                # Detect license plates within the car bounding box area
                car_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                plate_detections = model_plate(car_crop)[0]

                for i, plate in enumerate(plate_detections.boxes.data.tolist()):
                    px1, py1, px2, py2, plate_score, plate_class_id = plate
                    plate_box = [px1 + x1, py1 + y1, px2 + x1, py2 + y1]  # Adjust coordinates relative to the original frame

                    # Check if this plate is similar to any previously detected plates
                    is_new_plate = True
                    for prev_plate_box in previous_plate_boxes:
                        if calculate_iou(plate_box, prev_plate_box) > iou_threshold:
                            is_new_plate = False
                            break

                    # Save the plate image if it's a new detection
                    if is_new_plate:
                        # Crop the license plate from the frame
                        license_plate_crop = frame[int(plate_box[1]):int(plate_box[3]), int(plate_box[0]):int(plate_box[2])]
                        
                        # Save the cropped license plate image
                        cropped_path = os.path.join(output_dir, f'frame_{frame_nmr:04d}_plate_{i}.png')
                        cv2.imwrite(cropped_path, license_plate_crop)
                        print(f"Saved new plate image: {cropped_path}")
                        
                        # Add this plate's bounding box to the list of previous plates
                        previous_plate_boxes.append(plate_box)

# Release video capture object
cap.release()

print('Cropping completed and unique images saved in crop_img.')
