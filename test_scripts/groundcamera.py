"""
Test script takes a few pics with the ground camera
and downloads them locally to your computer from the drone.


** THIS SCRIPT REQUIRES A LOCAL FOLDER OF THE YOLOv5 REPO WITHIN THE 
   SAME FOLDER AS THIS PYTHON SCRIPT **

"""

from pyparrot.Minidrone import Mambo
import cv2
import sys
import os
import torch
from pathlib import Path
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def save_picture(mambo_object,pictureName,path,filename):

    storageFile = path +'\\'+ filename
    print('Your Mambo groundcam files will be stored here',storageFile)

    if (mambo_object.groundcam.ftp is None):
        print("No ftp connection")

    # otherwise return the photos
    mambo_object.groundcam.ftp.cwd(mambo_object.groundcam.MEDIA_PATH)
    try:
        mambo_object.groundcam.ftp.retrbinary('RETR ' + pictureName, open(storageFile, "wb").write) #download


    except Exception as e:
        print('error')

def is_in_the_list(l1,l2):
    for i in range(min(len(l1),len(l2))):
        if l1[i]!=l2[i]:
            return [i,l2[i]]
    return [len(l2)-1,l2[len(l2)-1]]

# Connect to the Mambo Drone
mambo = Mambo(None, use_wifi=True) #address is None since it only works with WiFi anyway
print("trying to connect to mambo now")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

list_of_images = []
picture_names_new = []

# Custom yolov5 model
model_path = "C:\\Users\\Mitchell\\OneDrive\\Desktop\\4th Year\\Capstone\\GroundCamera\\best.pt"
repo_path = "C:\\Users\\Mitchell\\OneDrive\\Desktop\\4th Year\\Capstone\\GroundCamera\\yolov5"
print(ROOT)
print(model_path)
print(repo_path)
model = torch.hub.load(repo_path,'custom', path=model_path, source='local')

# Process for drone to follow while connected
if (success):
    # get the state information
    mambo.flat_trim()
    print("sleeping")
    mambo.smart_sleep(1)
    mambo.ask_for_state_update()
    mambo.smart_sleep(1)
    mambo.safe_takeoff(5)
    mambo.hover()

    picture_names = mambo.groundcam.get_groundcam_pictures_names() #get list of availible files
    print(picture_names)
    print(len(picture_names))

    #reset the memory of the mambo
    if picture_names!=[]:
        for i in range(len(picture_names)):
            filename=picture_names[i]
            mambo.groundcam._delete_file(filename)

    #check if the pictures were deleted
    picture_names = mambo.groundcam.get_groundcam_pictures_names() #get list of availible files
    print(picture_names)
    print(len(picture_names))

     # take the photo
    picture_names = mambo.groundcam.get_groundcam_pictures_names()    #take picture
    mambo.hover()
    pic_success = mambo.take_picture()

    c = 0

    while True:
        # need to wait a bit for the photo to show up
        mambo.smart_sleep(0.5)

        picture_names = mambo.groundcam.get_groundcam_pictures_names()    #take picture

        mambo.smart_sleep(1)
        # mambo.hover()
        mambo.take_picture()
        mambo.smart_sleep(1)
        mambo.hover()    

        #mambo.smart_sleep(0.5)

        picture_names_new = mambo.groundcam.get_groundcam_pictures_names()#essential to reload it each time; does not update automaticaly
        print("New length is",len(picture_names_new))

        #finding the new picture in the list
        index=is_in_the_list(picture_names, picture_names_new)
        list_of_images.append(index[1])
        mambo.hover()

        print("Quick wait")

        #get the right picture
        picture_name=is_in_the_list(picture_names,picture_names_new)[1]
        print(picture_name)

        frame = mambo.groundcam.get_groundcam_picture(list_of_images[0],True) #get frame which is the first in the array
        path = sys.path[0]
        print(path)

        if frame is not None:
            print("GOT TO HERE")
            image = "test_image_%02d.png" % c
            save_picture(mambo, picture_name, path, image)
            image_path = ROOT / image
            results = model(image_path)

            for det in results.xyxy[0]:
                x_min, y_min, x_max, y_max, confidence, class_label = det
                if confidence > 0.70:
                    print("LANDING PAD SPOTTED!")

                    # Draw the bounding box
                    frame = cv2.imread(str(image_path))  # Read the image file
                    color = (0, 255, 0)  # Green color for the bounding box
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color,
                                  2)  # Draw the rectangle

                    # You may want to add text for the label as well
                    label = f'Confidence: {confidence:.2f}'
                    cv2.putText(frame, label, (int(x_min), int(y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Save or show the image
                    detected_image_path = str(ROOT / f"detected_{image}")
                    cv2.imwrite(detected_image_path, frame)  # Save the image with the bounding box
                    print(f"Saved detected image with bounding box to {detected_image_path}")

                    # If you want to display the image, uncomment the following line
                    # cv2.imshow('Detected Landing Pad', frame)
                    # cv2.waitKey(0)  # Wait for a key press to close the displayed image

                    mambo.safe_land(5)
                    mambo.disconnect()
                    break  # Exit the loop if landing pad spotted

            c = c + 1

        if c>20:
            break


    mambo.safe_land(5)
    mambo.disconnect()
