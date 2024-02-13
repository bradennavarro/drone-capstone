# Autonomous Indoor Mapping and Environmental Detection with Parrot Mambo Drone
Capstone project for ENGO 500. Members: Mitchell Aitken, Braden Navarro, Mac Dressler, Julian Cramb



## Overview
This repository contains the code for a capstone project focused on autonomous indoor mapping and environmental detection using a Parrot Mambo toy drone. The project's goal is to enable the drone to autonomously navigate indoor spaces and identify environmental elements using computer vision and machine learning techniques.

## Project Structure

### ENNEDetectionScripts
- `main.py`: The main script to execute environmental detection.
- `plant.pt` and `spill.pt`: Pre-trained PyTorch models for detecting plants and spills, respectively.
- `PlantImages` contains image data used for training the plant detection model.
- `SpillImages` contains image data used for training the spill detection model.

### WildLifeCounter
- `Wildlifecounter.py`: The main script. Uses digital imaging principles and CV2 to count objects to be used for simple wildlife monitoring.
- `processed_Mambo_1970-01-01T000325+0000_CB098B.jpg`: Example image of a succesful count.

### Image Stitching
- `imagestitching.py`: Script for stitching images to create a comprehensive map of the indoor environment.
- `ImageStitchingPhotosDrone`: Folder containing images taken from the drone for stitching.
- `ImageStitchingPhotosPhone`: Folder containing images taken from a phone for stitching.

### Test Scripts
- `object_detection_test`: Folder with scripts to test object detection functionality.
- `Minidrone.py`: Script that interfaces with the Parrot Mambo drone for basic flight control.
- `VLCFrontCam.py`: Script for capturing images from the drone's front camera.
- `groundcamera.py`: Updated script for capturing images with bounding box output for detected objects.

## Usage
Before running the scripts, ensure you have the required dependencies installed, downloading via pip. 
NOTE: Replace the Minidrone.py file in the PyParrot library with the Minidrone.py in this repo.



 ## Acknowledgments
- Special thanks to the contributors and advisors who have made this project possible. Namely our advisor Kyle O'Keefe and previous student Claire Mah.
- Acknowledgment to the open-source libraries such as PyParrot and tools used in this project.
 
