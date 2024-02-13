from pyparrot.Minidrone import Mambo
import cv2
import sys
import os
import numpy as np

def download_picture(mambo_object, pictureName, path):
    # saving the pic from the drone
    storageFile = os.path.join(path, pictureName)
    print('downloading to:', storageFile)

    if mambo_object.groundcam.ftp is None:
        print("no ftp :(")
        return None

    try:
        # try to get the pic
        mambo_object.groundcam.ftp.cwd(mambo_object.groundcam.MEDIA_PATH)
        mambo_object.groundcam.ftp.retrbinary('RETR ' + pictureName, open(storageFile, "wb").write)
        return storageFile
    except Exception as e:
        print('oops', e)
        return None

def detect_objects_and_draw_bounding_boxes(image):
    # make it gray and blurry
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # edges
    edged = cv2.Canny(blurred, 50, 150)

    # make edges thicker then thin them
    dilated = cv2.dilate(edged, None, iterations=2)
    eroded = cv2.erode(dilated, None, iterations=1)

    # find the shapes
    contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ignore tiny stuff
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 100]

    object_count = 0
    for contour in filtered_contours:
        # simplify the shapes
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # draw boxes around stuff
        x, y, w, h = cv2.boundingRect(approx)
        if w < image.shape[1] * 0.5 and h < image.shape[0] * 0.5:  # skip big boxes
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            object_count += 1

    # count stuff
    cv2.putText(image, f"count: {object_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image, object_count

mambo = Mambo(None, use_wifi=True)
print("connecting to mambo")
success = mambo.connect(num_retries=3)
print("connected?", success)

if success:
    mambo.flat_trim()
    mambo.smart_sleep(1)
    mambo.ask_for_state_update()
    mambo.smart_sleep(1)
    mambo.safe_takeoff(5)
    mambo.hover()

    # delete old pics
    picture_names = mambo.groundcam.get_groundcam_pictures_names()
    for filename in picture_names:
        mambo.groundcam._delete_file(filename)

    image_files = []

    # taking new pics
    for _ in range(6):
        mambo.take_picture()
        mambo.smart_sleep(2)

    picture_names_new = mambo.groundcam.get_groundcam_pictures_names()

    # get pics to computer
    path = sys.path[0]
    for picture_name in picture_names_new:
        image_file = download_picture(mambo, picture_name, path)
        if image_file:
            image_files.append(image_file)

    mambo.safe_land(5)
    mambo.disconnect()

    # find stuff in pics
    for image_file in image_files:
        image = cv2.imread(image_file)
        if image is not None:
            processed_image, object_count = detect_objects_and_draw_bounding_boxes(image)
            processed_filename = "done_" + os.path.basename(image_file)
            cv2.imwrite(os.path.join(path, processed_filename), processed_image)
            print(f"done {processed_filename} with {object_count} things")
