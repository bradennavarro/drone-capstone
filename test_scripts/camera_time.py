"""
Test script to test how long it takes to take pictures with ground cam
"""

from pyparrot.Minidrone import Mambo
import cv2
import time
import numpy as np
import os

def line_detection(mambo, image):
    
    # Get dimensions of image
    size = image.shape
    image_height = size[0]
    image_width = size[1]

    # Convert the image to grayscale
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  

    # Apply edge detection method on the image
    edges = cv2.Canny(grey, 50, 150, apertureSize=3)

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
            # Stores the value of cos(theta) in a
            a = np.cos(theta)
 
            # Stores the value of sin(theta) in b
            b = np.sin(theta)
 
            # x0 stores the value rcos(theta)
            x0 = a*r
 
            # y0 stores the value rsin(theta)
            y0 = b*r
 
            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000*(-b))
 
            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000*(a))
 
            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000*(-b))
 
            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000*(a))
 
            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            xm1 = xm1 + x1
            xm2 = xm2 + x2
            ym1 = ym1 + y1
            ym2 = ym2 + y2
    else:
        print("No angles found")
        return None

    xm1 = int(xm1 / num_lines)
    xm2 = int(xm2 / num_lines)
    ym1 = int(ym1 / num_lines)
    ym2 = int(ym2 / num_lines)
    try:
        m = (ym2-ym1)/(xm2-xm1) # Slope of line
    except:
        return None
    b = ym1 - m*xm1
    xlow = (0-b)/m
    xtop = (480-b)/m
    angle = np.arctan2(image_height,xtop-xlow) * (180/np.pi)

    # TEST to find middle of line 
    x_line = ((image_height/2)-b)/m
    x_diff = (image_width/2) - x_line # pixels
    print(f"Horizontal Adjustment: {x_diff / 6} m")

    mid_img_width = image_width/2

    if xlow > xtop:
        angle = -(90-angle)
        print(f"ANGLE: {angle}")
        #mambo.turn_degrees(angle)
    elif xlow < xtop:
        angle = angle-90
        print(f"ANGLE: {angle}")
        #mambo.turn_degrees(angle)

    # Plots central of the line
    cv2.line(image, (int(xlow), 0), (int(xtop), image_height), (0, 255, 0), 2)
    # Plots midway line of image
    cv2.line(image, (int(image_width/2),0), (int(image_width/2),image_height), (255, 0, 0), 2)
    cv2.line(image, (0,int(image_height/2)), (int(image_width),int(image_height/2)), (255, 0, 0), 2)

    # Specify the font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)  # Coordinates of the top right corner of the text
    font_scale = 1  # Font scale
    color = (255, 0, 0)  # Color in BGR format (white)
    thickness = 2  # Line thickness

    text = str(int(angle)) + "," + str(int((x_diff / 60)*10)) + "m"

    # Write the text onto the image
    cv2.putText(image, text, org, font, font_scale, color, thickness)

    return None

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
    mambo.smart_sleep(5)
    start_time = time.time()
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
    print(f"Pictures length: {len(picture_names)}")
    pic_success = mambo.take_picture()
    mambo.smart_sleep(0.2)
    picture_names = mambo.groundcam.get_groundcam_pictures_names()
    print(picture_names)
    try:
        image = mambo.groundcam.get_groundcam_picture(picture_names[0],True) # returns a cv2 image file
        cv2.imshow('Image',image)
        path = r'C:\Users\brade\OneDrive\Documents\CapstoneProject\ground_camera_test\image.jpg'
        cv2.imwrite(path, image)
    except:
        print("No image taken")
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ",execution_time, "seconds")

    #----------------------------------------------------------------------
    print("2nd TIME TEST")
    start_time = time.time()
    image_name=picture_names[0]
    mambo.groundcam._delete_file(image_name)
    mambo.smart_sleep(0.2)
    pic_success = mambo.take_picture()
    mambo.smart_sleep(0.2)
    picture_names = mambo.groundcam.get_groundcam_pictures_names()
    try:
        image = mambo.groundcam.get_groundcam_picture(picture_names[0],True) # returns a cv2 image file
        cv2.imshow('Image',image)
        images.append(image)
    except:
        print("No image taken")
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ",execution_time, "seconds")

    #----------------------------------------------------------------------
    print("3rd TIME TEST - Line Detection")
    start_time = time.time()
    image_name=picture_names[0]
    mambo.groundcam._delete_file(image_name)
    mambo.smart_sleep(0.2)
    pic_success = mambo.take_picture()
    mambo.smart_sleep(0.2)
    picture_names = mambo.groundcam.get_groundcam_pictures_names()
    try:
        image = mambo.groundcam.get_groundcam_picture(picture_names[0],True) # returns a cv2 image file
        line_detection(mambo,image)
        cv2.imshow('Image',image)
        images.append(image)
    except:
        print("No image taken")
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ",execution_time, "seconds")

    for i, processed_image in enumerate(images):
        filename = f'processed_image_{i}.jpg'
        cv2.imwrite(filename, processed_image)

    mambo.disconnect()