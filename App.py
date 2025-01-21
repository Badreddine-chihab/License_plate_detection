import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading

# Load YOLO models for detection tasks
model_car = YOLO('yolov8n.pt')  # Pre-trained YOLOv8 model for car detection
model_plate = YOLO('license_plate_model.pt')  # Custom YOLO model for license plate detection
model_char = YOLO('character_detection.pt')  # Custom YOLO model for character recognition

# Mapping dictionary for detected character labels to Arabic characters
label_mapping = {
    'a': 'أ',
    'b': 'ب',
    'd': 'د',
    'waw': 'و',
    'h': 'ه',
    'ch': 'ش',
}

# Function to process the uploaded image
def process_image(image_path):
    try:
        # Display "Processing..." message and start the loading animation
        result_text.set("Processing...")
        progress_bar.start()

        # Read the uploaded image using OpenCV
        frame = cv2.imread(image_path)

        # Detect cars in the image using the YOLO car model
        car_detections = model_car(frame)[0]
        for detection in car_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) == 2:  # Check if detected object is a car (class ID 2 for cars)
                # Crop the car region from the image
                car_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                # Detect license plates in the cropped car region
                plate_detections = model_plate(car_crop)[0]
                for plate_detection in plate_detections.boxes.data.tolist():
                    px1, py1, px2, py2, pscore, pclass_id = plate_detection

                    # Crop the detected license plate region
                    license_plate = car_crop[int(py1):int(py2), int(px1):int(px2)]

                    # Resize the license plate for better character detection
                    license_plate_resized = cv2.resize(license_plate, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

                    # Detect characters in the license plate
                    char_detections = model_char(license_plate_resized)[0]
                    char_results = []
                    for bbox, label in zip(char_detections.boxes.xyxy, char_detections.boxes.cls):
                        # Map detected character labels to their corresponding names
                        class_name = model_char.names[int(label)]
                        class_name = label_mapping.get(class_name, class_name)  # Map to Arabic character if applicable
                        char_results.append((int(bbox[0]), class_name))  # Append character with its x-coordinate

                    # Sort detected characters by their x-coordinate for proper sequence
                    char_results.sort(key=lambda x: x[0])
                    detected_text = ''.join([char[1] for char in char_results])  # Form the detected license plate text

                    # Update the result label with the detected license plate text
                    result_text.set(f"Detected Plate: {detected_text}")

                    # Display the cropped license plate in the output section
                    show_cropped_license_plate(license_plate_resized)
                    break  # Stop after processing the first detected license plate

        # Stop the loading animation
        progress_bar.stop()
        progress_bar.config(mode="determinate", value=0)
        if result_text.get() == "Processing...":
            result_text.set("No license plate detected.")  # If no license plate is detected, update the message
    except Exception as e:
        # Handle any errors during processing
        progress_bar.stop()
        progress_bar.config(mode="determinate", value=0)
        result_text.set("Error during processing.")  # Update the result text to indicate an error
        messagebox.showerror("Error", f"Processing failed: {e}")  # Show an error message box

# Function to display the cropped license plate
def show_cropped_license_plate(image):
    # Convert OpenCV image to a format compatible with Tkinter
    license_plate_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = license_plate_image.size

    # Scale the image to fit within a 200x100 area while maintaining the aspect ratio
    scale_factor = min(200 / width, 100 / height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    license_plate_image = license_plate_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Display the resized license plate in the output label
    license_plate_image = ImageTk.PhotoImage(license_plate_image)
    output_label.config(image=license_plate_image)
    output_label.image = license_plate_image  # Keep a reference to avoid garbage collection

# Function to load an image from the file system
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        # Open the selected image and resize it for display in the input label
        img = Image.open(file_path)
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        input_label.config(image=img)
        input_label.image = img  # Keep a reference to avoid garbage collection

        # Process the image in a separate thread to prevent the GUI from freezing
        threading.Thread(target=process_image, args=(file_path,)).start()

# Create the main Tkinter window
root = tk.Tk()
root.title("License Plate Detection")  # Set the window title
root.geometry("800x800")  # Set the window size
root.configure(bg="#f7f7f7")  # Set a light background color for a modern look

# Header label with a larger font for the title
header_label = tk.Label(root, text="Moroccan License Plate Detection", font=("Helvetica", 24, "bold"), bg="#f7f7f7", fg="#333")
header_label.pack(pady=10)

# Frame for input and output image sections
frame = ttk.Frame(root)
frame.pack(pady=10)

# Input image section with a labeled frame
input_label_frame = ttk.LabelFrame(frame, text="Select your frame here", padding=(10, 10))
input_label_frame.grid(row=0, column=0, padx=10, pady=10)

# Placeholder label for displaying the input image
input_label = tk.Label(input_label_frame, bg="#e6e6e6")
input_label.pack()

# Button to load an image
load_button = ttk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

# Result section with a labeled frame for displaying the detected license plate text
result_frame = ttk.LabelFrame(root, text="Detected Plate", padding=(10, 10))
result_frame.pack(pady=10)

# Label to display the detected license plate text
result_text = tk.StringVar()
result_text.set("Detected Plate: None")
result_label = tk.Label(result_frame, textvariable=result_text, font=("Helvetica", 16), bg="#f7f7f7")
result_label.pack()

# Output image section with a labeled frame for displaying the cropped license plate
output_label_frame = ttk.LabelFrame(frame, text="Cropped Plate", padding=(10, 10))
output_label_frame.grid(row=0, column=1, padx=10, pady=10)

# Placeholder label for displaying the cropped license plate
output_label = tk.Label(output_label_frame, bg="#e6e6e6")
output_label.pack()

# Progress bar for showing a loading animation during processing
progress_bar = ttk.Progressbar(root, mode="indeterminate", length=400)
progress_bar.pack(pady=10)

# Status bar for additional user feedback
status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#f7f7f7", fg="#333")
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Apply modern styles to the UI elements
style = ttk.Style()
style.configure("TLabelFrame", background="#f7f7f7", font=("Helvetica", 12))
style.configure("TButton", font=("Helvetica", 12), background="#d9d9d9", foreground="#333")
style.map("TButton", background=[("active", "#c6c6c6")])

# Start the Tkinter event loop
root.mainloop()
