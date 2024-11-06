import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('character_detection.pt')  # Path to your saved model

# Paths for input images and output results
images_path = './crop_img'  # Folder containing input images
output_im_path = './results'  # Folder to save processed images with detections

# Mapping dictionary
label_mapping = {
    'a': 'أ',
    'b': 'ب',
    'd': 'د',
    'waw': 'و',
    'h': 'ه',
    'ch': 'ش',
}

# Ensure output directory exists
os.makedirs(output_im_path, exist_ok=True)

# Function to detect objects in an image and print labels in left-to-right order
def detect_and_order_labels(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Perform inference (object detection) using the model
    results = model(img)  # Get predictions (results is a list of Result objects)
    result = results[0]  # Get the first result from the list

    # Extract bounding boxes and class labels
    detections = []
    for bbox, label in zip(result.boxes.xyxy, result.boxes.cls):
        x1, y1, x2, y2 = map(int, bbox)  # Get coordinates of the bounding box
        class_id = int(label)  # Convert label to integer
        class_name = model.names[class_id]  # Get the class name using the index
        
        # Map label to Arabic if it exists in the dictionary
        class_name = label_mapping.get(class_name, class_name)
        
        # Append bounding box and label to detections
        detections.append((x1, y1, x2, y2, class_name))

    # Sort detections by x1 (left coordinate) to maintain left-to-right order
    detections.sort(key=lambda x: x[0])  # Sort by x1 to order from left to right

    # Get ordered labels and print them
    ordered_labels = [det[4] for det in detections]  # Extract only the labels in order
    print(f"Image: {os.path.basename(image_path)} - Detected labels (left-to-right): {''.join(ordered_labels)}")

    # Draw bounding boxes and labels on the image
    for x1, y1, x2, y2, class_name in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Put label

    # Save the image with bounding boxes and labels
    output_image_path = os.path.join(output_im_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, img)
    print(f"Detection results saved to: {output_image_path}")

    return img, ordered_labels

# Function to load images from a folder and process each
def load_and_process_images(folder_path):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    processed_images = []
    all_labels = []

    # Process each image file
    for image_path in image_files:
        img, labels = detect_and_order_labels(image_path)
        processed_images.append(img)
        all_labels.append((os.path.basename(image_path), labels))

    return processed_images, all_labels

# Load and process all images from the specified folder
processed_images, all_labels = load_and_process_images(images_path)

# Display results in a grid
def display_images_grid(images, labels, cols=3):
    rows = (len(images) + cols - 1) // cols  # Calculate number of rows needed for the grid
    plt.figure(figsize=(15, rows * 5))
    
    for i, (img, (image_name, label)) in enumerate(zip(images, labels)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{image_name}\nLabels: {''.join(label)}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Display all processed images in a grid
display_images_grid(processed_images, all_labels, cols=3)
