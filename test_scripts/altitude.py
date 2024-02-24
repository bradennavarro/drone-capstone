from pyparrot.Minidrone import Mambo
import cv2
import time
import numpy as np
import os

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
    altitude = mambo.sensors.altitude
    euler_angle = mambo.sensors.quaternion_to_euler_angle(mambo.sensors.quaternion_w,
                                                          mambo.sensors.quaternion_x,
                                                          mambo.sensors.quaternion_y,
                                                          mambo.sensors.quaternion_z)
    print("ALTITUDE: " + str(altitude))
    print("ANGLE: " + str(euler_angle))

    print("------ TAKEOFF -----")
    mambo.safe_takeoff(5)
    mambo.smart_sleep(2)
    mambo.hover()
    altitude = mambo.sensors.altitude
    euler_angle = mambo.sensors.quaternion_to_euler_angle(mambo.sensors.quaternion_w,
                                                          mambo.sensors.quaternion_x,
                                                          mambo.sensors.quaternion_y,
                                                          mambo.sensors.quaternion_z)
    print("ALTITUDE: " + str(altitude))
    print("ANGLE: " + str(euler_angle))

    print("------ FLY -----")
    mambo.fly_direct(-100,0,0,0,0.5)
    mambo.smart_sleep(2)

    print("------ TURN -----")
    mambo.turn_degrees(90)
    mambo.smart_sleep(2)
    mambo.hover()
    altitude = mambo.sensors.altitude
    euler_angle = mambo.sensors.quaternion_to_euler_angle(mambo.sensors.quaternion_w,
                                                          mambo.sensors.quaternion_x,
                                                          mambo.sensors.quaternion_y,
                                                          mambo.sensors.quaternion_z)
    print("ALTITUDE: " + str(altitude))
    print("ANGLE: " + str(euler_angle))

    print("------ UP 0.5 m -----")
    mambo.fly_direct(0,0,0,100,0.5)
    mambo.smart_sleep(2)
    mambo.hover()
    altitude = mambo.sensors.altitude
    euler_angle = mambo.sensors.quaternion_to_euler_angle(mambo.sensors.quaternion_w,
                                                          mambo.sensors.quaternion_x,
                                                          mambo.sensors.quaternion_y,
                                                          mambo.sensors.quaternion_z)
    print("ALTITUDE: " + str(altitude))
    print("ANGLE: " + str(euler_angle))

    print("------ DOWN 1 m -----")
    mambo.fly_direct(0,0,0,-100,1)
    mambo.smart_sleep(2)
    mambo.hover()
    altitude = mambo.sensors.altitude
    euler_angle = mambo.sensors.quaternion_to_euler_angle(mambo.sensors.quaternion_w,
                                                          mambo.sensors.quaternion_x,
                                                          mambo.sensors.quaternion_y,
                                                          mambo.sensors.quaternion_z)
    print("ALTITUDE: " + str(altitude))
    print("ANGLE: " + str(euler_angle))

    alt_diff = 1 - altitude
    print("------ DOWN " + str(alt_diff) + "m -----")
    if alt_diff >= 0:
        mambo.fly_direct(0,0,0,100,alt_diff)
    else:
        mambo.fly_direct(0,0,0,-100,-alt_diff)
    mambo.smart_sleep(2)
    mambo.hover()
    altitude = mambo.sensors.altitude
    euler_angle = mambo.sensors.quaternion_to_euler_angle(mambo.sensors.quaternion_w,
                                                          mambo.sensors.quaternion_x,
                                                          mambo.sensors.quaternion_y,
                                                          mambo.sensors.quaternion_z)
    print("ALTITUDE: " + str(altitude))
    print("ANGLE: " + str(euler_angle))
    


    mambo.safe_land(5)

    print("BATTERY: " + str(mambo.sensors.battery))
    mambo.disconnect()