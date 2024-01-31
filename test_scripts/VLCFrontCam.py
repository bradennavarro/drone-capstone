"""
Demo of the Bebop vision using DroneVisionGUI that relies on libVLC.  It is a different
multi-threaded approach than DroneVision

Author: Amy McGovern
"""
from pyparrot.Minidrone import Mambo
from pyparrot.DroneVisionGUI import DroneVisionGUI
from pyparrot.Minidrone import Mambo
import cv2
import sys
import os
import torch
from pathlib import Path
from PIL import Image
import cv2



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
#ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



# set this to true if you want to fly for the demo
testFlying = True

# Custom yolov5 model
model_path = ROOT / 'best.pt'
repo_path = ROOT / 'yolov5'
print(ROOT)
print(model_path)
print(repo_path)
model = torch.hub.load(r'C:\Users\jlncr\Documents\4_year\ENGO_500\DroneTesting\yolov5','custom', path=r'C:\Users\jlncr\Documents\4_year\ENGO_500\DroneTesting\best.pt', source='local')

class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        # print("in save pictures on image %d " % self.index)

        img = self.vision.get_latest_valid_picture()

        if (img is not None):
            filename = "test_image_%06d.png" % self.index
            # uncomment this if you want to write out images every time you get a new one
            #cv2.imwrite(filename, img)
            self.index +=1
            #print(self.index)


def demo_mambo_user_vision_function(mamboVision, args):
    """
    Demo the user code to run with the run button for a mambo

    :param args:
    :return:
    """
    mambo = args[0]

    if (testFlying):
        print("taking off!")
        mambo.safe_takeoff(5)



        if (mambo.sensors.flying_state != "emergency"):
            print("flying state is %s" % mambo.sensors.flying_state)

            for ind in range(100):
                image = "C:\\Users\\jlncr\\.conda\\envs\\DroneTesting\\Lib\\site-packages\\pyparrot\\images\\visionStream.jpg"
                print(ind)
                results = model(image)
                for det in results.xyxy[0]:
                    x_min, y_min, x_max, y_max, confidence, class_label = det
                    if confidence > 0.70:
                        print("LANDING PAD SPOTTED!")
                        results.save()
                        mambo.safe_land(5)
                        mambo.disconnect()
                        break

            # mambo.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=2)
            mambo.smart_sleep(5)

        print("landing")
        print("flying state is %s" % mambo.sensors.flying_state)
        mambo.safe_land(5)
    else:
        print("Sleeeping for 15 seconds - move the mambo around")
        mambo.smart_sleep(15)

    # done doing vision demo
    print("Ending the sleep and vision")
    mamboVision.close_video()

    mambo.smart_sleep(5)

    print("disconnecting")
    mambo.disconnect()


if __name__ == "__main__":
    # you will need to change this to the address of YOUR mambo
    mamboAddr = "e0:14:d0:63:3d:d0"

    # make my mambo object
    # remember to set True/False for the wifi depending on if you are using the wifi or the BLE to connect
    mambo = Mambo(mamboAddr, use_wifi=True)
    print("trying to connect to mambo now")
    success = mambo.connect(num_retries=3)
    print("connected: %s" % success)

    if (success):
        # get the state information
        print("sleeping")
        mambo.smart_sleep(1)
        mambo.ask_for_state_update()
        mambo.smart_sleep(1)

        print("Preparing to open vision")
        mamboVision = DroneVisionGUI(mambo, is_bebop=False, buffer_size=200,
                                     user_code_to_run=demo_mambo_user_vision_function, user_args=(mambo, ))
        userVision = UserVision(mamboVision)
        mamboVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)

        mamboVision.open_video()

        image = "C:\\Users\\jlncr\\.conda\\envs\\DroneTesting\\Lib\\site-packages\\pyparrot\\images\\visionstream.jpg"
        results = model(image)
        for det in results.xyxy[0]:
            x_min, y_min, x_max, y_max, confidence, class_label = det
            if confidence > 0.70:
                print("LANDING PAD SPOTTED!")
                results.save()
                mambo.safe_land(5)
                mambo.disconnect()
                break