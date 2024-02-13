# Wildlife Counter #

`WildlifeCounter.py` is a Python script designed to help in counting wildlife, specifically aimed at monitoring migrating deer in grassy plains. Using drone-captured images, this script processes each photo to detect deer, draw bounding boxes around each identified animal, and count the total number in the frame.

## Features

- **Image Processing**: Converts drone-captured images to grayscale, applies edge detection, and uses contour detection to identify potential wildlife.
- **Bounding Boxes**: Draws green rectangles around detected wildlife to help users visually confirm detections.
- **Count Display**: Adds a count of detected wildlife to the top-left corner of each processed image.

## Requirements

To run `WildlifeCounter.py`, you'll need the following:

- Python
- OpenCV library (cv2)
- NumPy
- pyparrot library (for interacting with Parrot drones)

## Setup
1. Ensure Python 3.x is installed on your system.
2. Install the required Python libraries
3. Connect your drone to the Mambo Wifi Network
4. The script will automatically connect to the drone, take a series of images, and process each image to count and mark detected objects (wildlife)

## Note

`WildlifeCounter.py` is optimized for plain environments and may need adjustments for different terrains. The accuracy of wildlife detection can vary based on lighting conditions, drone altitude, and the size of the objects.

