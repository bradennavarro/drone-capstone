import numpy as np
import cv2
import glob


# compute ssim
def compute_ssim(img1, img2):
    # Constants for SSIM calculation
    C1 = (0.01 * 255) ** 2  # used to stabilize division by a small denominator
    C2 = (0.03 * 255) ** 2  # used in the luminance and contrast comparison functions

    # convert images to float64 to avoid overflow or underflow losses (idk if this is needed)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Gaussian kernel for SSIM computation; this is part of the low-pass filter
    # simulates the human visual system's multi-scale nature
    kernel = cv2.getGaussianKernel(11, 1.5)  # 1D Gaussian kernel
    window = np.outer(kernel, kernel.transpose())  # 2D Gaussian window by taking the outer product

    #  mean (mu) of img1 and img2 using the Gaussian window (low-pass filtering)
    # gives a weighted average around each pixel (local neighborhood)
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # Apply Gaussian filter and trim edges to avoid boundary effects
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]  # Same for img2

    # square of the means
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute the variance (sigma^2) of img1 and img2 using the Gaussian window
    # mean of the squared image minus the squared mean
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq

    # Compute the covariance (sigma12) between img1 and img2
    # mean of the product of images minus the product of means
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    # SSIM index map (pixel-wise SSIM values)
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # returns the average SSIM over all pixels, providing a single similarity measure between the two images
    return ssim_map.mean()



# evaluate overlap quality for images
def evaluate_sequence_overlap_quality(images):
    # Placeholder for results
    psnr_results = []
    ssim_results = []

    for i in range(len(images) - 1):
        # "assuming whole image is the overlap"
        overlap1 = images[i]
        overlap2 = images[i + 1]

        # resize overlap2 to match overlap1's dimensions (for some reason they weren't the exact same size)
        overlap2_resized = cv2.resize(overlap2, (overlap1.shape[1], overlap1.shape[0]))

        # PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index)
        # for the overlap regions between consecutive images in your stitched panorama.
        psnr = cv2.PSNR(overlap1, overlap2_resized) #cv2 has a PSNR function
        ssim = compute_ssim(overlap1, overlap2_resized)

        psnr_results.append(psnr)
        ssim_results.append(ssim)

    return psnr_results, ssim_results



# Load images from the specified folder (this is just hardcoded for now)
folder_path = 'C:\\Users\\Mitchell\\OneDrive\\Desktop\\4th Year\\Capstone\\Image Stitching\\Image Stitching\\ImageStitchingPhotosDrone\\'
image_paths = glob.glob(folder_path + '*.png')
images = [cv2.imread(image) for image in image_paths]

# Print the loaded files
print("Loaded files:")
for image_path in image_paths:
    print(image_path)

# resize images to a common height for stitching
common_height = 600
images_resized = [cv2.resize(img, (int(img.shape[1] * common_height / img.shape[0]), common_height)) for img in images]
print("Images resized (bigger seems to mess it up)")


# Fucntion to Stitch images using RANSAC and homography
def stitch_images(images):
    stitcher = cv2.Stitcher_create()  # Stitcher object

    status, stitched_img = stitcher.stitch(images)  # Stitch images

    if status == cv2.Stitcher_OK:
        return stitched_img
    else:
        print("Error during stitching")
        return None


# I didn't use this but found in YT video, crops the black border
def crop_black_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return image[y:y + h, x:x + w]


# call
print("Calling image stitching algorithm")
stitched_image = stitch_images(images_resized)

if stitched_image is not None:
    # Crop black borders
    stitched_image = crop_black_borders(stitched_image)

    # Save the stitched image
    output_path = 'C:\\Users\\Mitchell\\OneDrive\\Desktop\\4th Year\\Capstone\\Image Stitching\\Image Stitching\\ImageStitchingPhotosPhone\\stitched_Images\\stichedOutputDroneTest.png'
    cv2.imwrite(output_path, stitched_image)
    print("Wrote to output path")

    # Display the stitched image
    cv2.imshow("Stitched Image", stitched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Stitched image saved to: {output_path}")

    # Evaluate the stitching quality here
    psnr_results, ssim_results = evaluate_sequence_overlap_quality([stitched_image] + images_resized)
    for i, (psnr, ssim) in enumerate(zip(psnr_results, ssim_results)):
        print(f"Overlap Region {i}-{i + 1}: PSNR={psnr}, SSIM={ssim}")
else:
    print("Images could not be stitched!")
