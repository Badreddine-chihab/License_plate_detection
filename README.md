Moroccan License Plate Detection and Reader
This project is a comprehensive solution for detecting and reading Moroccan license plates. It uses custom datasets and machine learning models to detect license plates, segment characters, and translate them into Arabic characters.

Features and Components
Detection
Custom Dataset Preparation: The license plate detection is based on a dataset labeled using the labelImg tool. Data augmentation techniques were applied to enhance the model's robustness.
License Plate Detection Model: A YOLOv8-based model is used to detect license plates from images or frames extracted from videos.
Reader: The character recognition model classifies characters into one of 16 classes, transforming the detected license plates into readable Arabic text.
Data Sources
Character Recognition Dataset: The dataset for character recognition is sourced from the UM6P dataset.
Output Example: For a detected license plate like 26178WAW, the output is transformed into 2617و8 using a mapping process to convert Latin characters into their corresponding Arabic characters.
Workflow
1. Frame Extraction and Cropping
The project includes two scripts that handle video and image processing:

cropping_vid.py:
Processes a video file by extracting frames.
Detects unique license plates within the frames.
Crops the detected plates and saves them in the crop_img directory.
Recommended for videos with shorter durations to ensure efficient processing.
cropimg.py:
Processes all images in a given directory to detect license plates.
Crops the license plates and saves them in the crop_img directory for further analysis.
2. License Plate Reading
Character Segmentation:
The cropped license plates saved in the crop_img directory are processed by a second YOLOv8-based model for character segmentation. This model:
Detects individual characters on the license plate.
Classifies the characters into one of 16 predefined classes.
Arabic Character Mapping:
The detected Latin-style characters are mapped back to their corresponding Arabic forms using a character mapping dictionary.
3. Visualization and Results
Visualization with Matplotlib:

The results are plotted in a grid layout using matplotlib.pyplot. Each plot displays the processed license plate with the detected characters.
Console Output:

The detected characters are printed in the console, with the Latin-to-Arabic character mapping applied for readability.
Results Directory:

All output images with detected characters are saved in the results directory for reference.
Application
App.py:
A GUI-based application that integrates the entire workflow for ease of use. The application provides:
Video and image upload functionalities.
Real-time processing of frames or images.
Detection results displayed within the application.
License plate translation into Arabic characters.
