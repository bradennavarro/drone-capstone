"""
Demo of the groundcam
Mambo takes off, takes a picture and shows a RANDOM frame, not the last one
Author: Valentin Benke, https://github.com/Vabe7
Author: Amy McGovern
"""

from pyparrot.Minidrone import Mambo
import cv2
import sys

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

mambo = Mambo(None, use_wifi=True) #address is None since it only works with WiFi anyway
print("trying to connect to mambo now")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)

list_of_images = []
picture_names_new = []

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
            filename = "test_image_%02d.png" % c
            save_picture(mambo,picture_name,path,filename)
            c = c+1

        if c>5:
            break


    mambo.safe_land(5)
    mambo.disconnect()