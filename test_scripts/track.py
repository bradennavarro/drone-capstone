"""
Test script to track and follow a line

IMPORTANT: For this file to run please clone the following repo
git clone https://github.com/ultralytics/yolov5.git
This repo should be saved in the same folder as this file 

WORK IN PROGRESS!
- Aims to fly around 1m
- Assuming it's following a straight line
- Assumes there are no other lines that will be detected

"""

from pyparrot.Minidrone import Mambo
import cv2
import math
import numpy as np
import os
import time
import torch

def altitude_correction(mambo):
    # Corrects altitude of drone to remain around 1m
    altitude = mambo.sensors.altitude
    alt_diff = 1 - altitude

    if abs(alt_diff) < 0.5:
        return

    if alt_diff >= 0:
        mambo.fly_direct(0,0,0,100,abs(alt_diff))
    else:
        mambo.fly_direct(0,0,0,-100,abs(alt_diff))
    mambo.smart_sleep(1)

def detect_landing_pad(mambo, model, image):

    cwd = os.getcwd()
    print(cwd)
    print(os.path.join(cwd,'best.pt'))

    size = image.shape
    image_height = size[0]
    image_width = size[1]

    results = model(image)
    detections = results.xyxy[0]

    if len(detections) == 0:
        return False
    else:
        for det in detections:
            x_min, y_min, x_max, y_max, confidence, class_label = det

    if confidence < 0.70:   
        return False

    # Draw the bounding box
    landing_pad = image  # Read the image file
    color = (0, 255, 0)  # Green color for the bounding box
    cv2.rectangle(landing_pad, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color,
                                  2)  # Draw the rectangle

    x = int((x_max + x_min)/2)
    y = int((y_max + y_min)/2)

    # Distance in pixels to landing pad
    pix_dist = math.sqrt((x-(image_width/2))**2+(y-(image_height/2))**2)
    # Distance in meters to landing pad
    dist = pix_dist / 635 # [m]

    angle = math.atan2(y-(image_height/2),x-(image_width/2))
    angle = math.degrees(angle) + 90 # To account for different frames

    cv2.rectangle(landing_pad, (int(x-2), int(y-2)), (int(x+2), int(y+2)), (0,0,255), 2)
    cv2.line(landing_pad, (int(x), int(y)), (int(image_width/2),int(image_height/2),), (0,0,255), 2) 

    # You may want to add text for the label as well
    label = f'Confidence: {confidence:.2f}'
    cv2.putText(landing_pad, label, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Plots central line of the image
    cv2.line(landing_pad, (int(image_width/2),0), (int(image_width/2),image_height), (255, 0, 0), 2)
    cv2.line(landing_pad, (0,int(image_height/2)), (int(image_width),int(image_height/2)), (255, 0, 0), 2)

    # Save or show the image
    detected_image_path = os.path.join(cwd,"detected_landing_pad.png")
    cv2.imwrite(detected_image_path, landing_pad)  # Save the image with the bounding box

    # Movement
    mambo.turn_degrees(angle)
    mambo.smart_sleep(0.5)
    mambo.fly_direct(0,30,0,0,dist)
    mambo.smart_sleep(2)

    return True


def line_detection(mambo, image):
    # Get dimensions of image
    size = image.shape
    image_height = size[0]
    image_width = size[1]

    # Convert the image to grayscale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    blur = cv2.GaussianBlur(grey,(5,5),0)

    _, thresholded_image = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Apply edge detection method on the image
    edges = cv2.Canny(thresholded_image, 50, 150, apertureSize=3)

    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 130)

    xm1 = 0
    xm2 = 0
    ym1 = 0
    ym2 = 0
    num_lines = 0

    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    if lines is not None:
        for r_theta in lines:
            num_lines += 1
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr

            a = np.cos(theta)
            b = np.sin(theta)
 
            x0 = a*r
            y0 = b*r
 
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
 
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            xm1 = xm1 + x1
            xm2 = xm2 + x2
            ym1 = ym1 + y1
            ym2 = ym2 + y2
    else:
        print("No lines found")
        return False

    # Average coordinates of line
    xm1 = int(xm1 / num_lines)
    xm2 = int(xm2 / num_lines)
    ym1 = int(ym1 / num_lines)
    ym2 = int(ym2 / num_lines)
    try:
        m = (ym2-ym1)/(xm2-xm1) # Slope of line
    except:
        return False
    b = ym1 - m*xm1 # y-intercept
    xlow = (0-b)/m
    xtop = (480-b)/m
    angle = np.arctan2(image_height,xtop-xlow) * (180/np.pi)

    # Find difference in x between middle of image and line
    x_line = ((image_height/2)-b)/m
    x_diff = (image_width/2) - x_line # pixels
    print("X_DIFF: " + str(x_diff))
    print(f"Horizontal Adjustment: {x_diff / 635} m")

    # Rotates drone to become parallel with line
    if xlow > xtop:
        angle = -(90-angle)
        print(f"Angle: {angle} degrees")
        mambo.turn_degrees(angle)
    elif xlow < xtop:
        angle = angle-90
        print(f"Angle: {angle} degrees")
        mambo.turn_degrees(angle)

    mambo.smart_sleep(1)

    # Adjusts drone to move horizontal towards the line
    #if x_diff >= 0:
    #    mambo.fly_direct(-100,0,0,0,(x_diff / 635))
    #else:
    #    mambo.fly_direct(100,0,0,0,(x_diff / 635))
    #mambo.smart_sleep(0.2)
    

    # Plots central of the line
    cv2.line(image, (int(xlow), 0), (int(xtop), image_height), (0, 255, 0), 2)
    # Plots midway line of image
    cv2.line(image, (int(image_width/2),0), (int(image_width/2),image_height), (255, 0, 0), 2)
    cv2.line(image, (0,int(image_height/2)), (int(image_width),int(image_height/2)), (255, 0, 0), 2)

    # Specify the font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (image.shape[1] - 150, 50)  # Coordinates of the top right corner of the text
    font_scale = 1  # Font scale
    color = (255, 0, 0)  # Color in BGR format 
    thickness = 2  # Line thickness

    text = str(int(angle)) + ", " + str(int(x_diff / 600)*10) + "cm"

    # Write the text onto the image
    cv2.putText(image, text, org, font, font_scale, color, thickness)

    return True


# Initialize YOLOv5 model
cwd = os.getcwd()
model_path = os.path.join(cwd,'best.pt')
repo_path = os.path.join(cwd,'yolov5')
model = torch.hub.load(repo_path,'custom', path=model_path, source='local')

# Images list
images = []

# Connect to drone
mambo = Mambo(None, use_wifi=True) #address is None since it only works with WiFi anyway
print("Attempting to connect to mambo...")
success = mambo.connect(num_retries=3)
print("Connected: %s" % success)

if (success):
    # get the state information
    print("sleeping")
    mambo.flat_trim()
    mambo.smart_sleep(1)
    mambo.ask_for_state_update()
    mambo.smart_sleep(2)

    picture_names = mambo.groundcam.get_groundcam_pictures_names()
    print(picture_names)

    #reset the memory of the mambo
    if picture_names!=[]:
        for i in range(len(picture_names)):
            filename=picture_names[i]
            mambo.groundcam._delete_file(filename)

    #check if the pictures were deleted
    picture_names = mambo.groundcam.get_groundcam_pictures_names() #get list of availible files
    print(picture_names)

    mambo.safe_takeoff(5)
    print("BATTERY: " + str(mambo.sensors.battery))
    mambo.hover()

    attempts = 0

    # Main tracking loop
    while (attempts < 3):

        start_time = time.time()

        # Correct altitude of drone
        #altitude_correction(mambo)

        mambo.take_picture()
        mambo.smart_sleep(0.2)

        try:
            picture_names = mambo.groundcam.get_groundcam_pictures_names()
            image = mambo.groundcam.get_groundcam_picture(picture_names[0],True) # returns a cv2 image file

            # Landing pad detection
            status = detect_landing_pad(mambo, model, image)
            if status:
                break
            #else:
            #    attempts = attempts + 1

            # Line detection
            line = line_detection(mambo,image)
            images.append(image)

            # Move drone forward
            if line:
                mambo.fly_direct(0,10,0,0,3)
                mambo.smart_sleep(1)
                mambo.hover()
            else:
                attempts += 1

            # Delete image from drone after use
            mambo.groundcam._delete_file(picture_names[0])
            mambo.smart_sleep(0.2)
        except:
            print("No image taken")
            continue
        end_time = time.time()
        #execution_time = end_time - start_time
        #print("Execution time: ",execution_time, "seconds")
        print(attempts)

    mambo.safe_land(5) 

    # Outputs the processed images
    for i, processed_image in enumerate(images):
        filename = f'processed_image_{i}.jpg'
        cv2.imwrite(filename, processed_image)  

    mambo.disconnect()
 