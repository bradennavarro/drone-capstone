import numpy as np
import cv2
import glob

#load images from where they are (can do not hard coded but I keep changing the folder)
image_paths = glob.glob(
    'C:\\Users\\Mitchell\\OneDrive\\Desktop\\4th Year\\Capstone\\Image Stitching\\ImageStitchingPhotosPhone\\from phone\\*.jpg')
images = [cv2.imread(image) for image in image_paths]

#resize images to a common height for stitching (seems like without this it gets an error saying it's too big sometimes)
common_height = 600
images_resized = [cv2.resize(img, (int(img.shape[1] * common_height / img.shape[0]), common_height)) for img in images]


# stitch images using RANSAC and homography
def stitch_images(images):
    stitcher = cv2.Stitcher_create()  # stitcher object

    status, stitched_img = stitcher.stitch(images) # stitch images

    if status == cv2.Stitcher_OK:
        return stitched_img
    else:
        print("Error during stitching")
        return None


# optional function crop black borders from the stitched image
def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return image[y:y + h, x:x + w]

# call function to stitch images
stitched_image = stitch_images(images_resized)

if stitched_image is not None:
    # crop black borders
    stitched_image = crop_black_borders(stitched_image)

    # save the stitched image in the same directory (or a different one if wanted)
    output_path = 'C:\\Users\\Mitchell\\OneDrive\Desktop\\4th Year\\Capstone\Image Stitching\\ImageStitchingPhotosPhone\\stitched_Images\\stichedOutput.png'
        #will have to modify this eventually to have multiple file names too but for now just manual
    cv2.imwrite(output_path, stitched_image)

    # display the stitched image
    cv2.imshow("Stitched Image", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Stitched image saved to: {output_path}")
else:
    print("Images could not be stitched!")
