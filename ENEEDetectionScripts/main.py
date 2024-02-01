
#https://universe.roboflow.com/6702-dakshayini-yaa1o/weedetection/dataset/4

#other potential ones?

#https://universe.roboflow.com/augmented-startups/weeds-nxe1w
# https://github.com/grimmlab/UAVWeedSegmentation?tab=readme-ov-file

#Spill model https://universe.roboflow.com/latifa-rdsiv/chemical-spills_


import cv2
import os
import torch
from PIL import Image

# Load the trained YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='spill.pt')  # Adjust the path as needed

# Set the confidence threshold
model.conf = 0.1  # Set the confidence threshold to 10%

def detect_image(image_path):
    print(f"Processing {image_path}...")  # Diagnostic print

    # Load image
    image = Image.open(image_path)

    # Perform detection
    results = model(image)

    # Print results to console
    results.print()  # Print results to console

    if len(results.xyxy[0]) == 0:
        print("No detections or below confidence threshold.")
    else:
        # Extracting results
        detected_objects = results.xyxy[0]  # x1, y1, x2, y2, confidence, class
        print(f"Detected objects in {os.path.basename(image_path)}:")
        for *box, conf, cls in detected_objects:
            label = model.names[int(cls)]  # Get the class label
            conf_percent = f"{conf * 100:.2f}%"  # Format confidence as a percentage
            print(f"Class: {label}, Confidence: {conf_percent}, BBox: {box}")

    # Try to display the image with bounding boxes
    try:
        results.show()
    except Exception as e:
        print(f"Unable to display image: {e}")  # Diagnostic print in case of an error

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):  # Added '.webp' to the supported formats
            image_path = os.path.join(directory_path, filename)
            detect_image(image_path)
            print(f"Processed {filename}")

if __name__ == "__main__":
    directory_path = r'C:\Users\mitch\OneDrive\Desktop\Uni\Capstone\ENEEPlantScript\Spillimages'  # Your image directory
    process_directory(directory_path)
