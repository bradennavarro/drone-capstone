# Image Stitching for Parrot Mambo MiniDrone Images #

This script is designed to stitch together a sequence of images taken from a Parrot Mambo MiniDrone. It utilizes OpenCV to perform the stitching, with additional functionality to evaluate the quality of the stitched images using metrics like PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).

## Requirements

- Python
- OpenCV (`cv2` library)
- NumPy
- Glob

Make sure you have the required libraries installed. You can install them using pip:

\`\`\`bash
pip install numpy opencv-python glob2
\`\`\`

## Usage

1. Place your sequence of images in a designated folder.
2. Modify the \`folder_path\` variable in the script to point to your images' directory.

The script will:
- Load all PNG images from the specified folder.
- Resize the images to a common height for consistency.
- Stitch the images together using OpenCV's \`Stitcher_create\` function.
- Optionally, crop out black borders for a cleaner look.
- Save the stitched image to a specified output path.
- Display the stitched image.
- Evaluate and print the PSNR and SSIM for the overlap regions between consecutive images.

## Customization

You can customize the script by modifying:
- \`folder_path\`: Directory containing your images.
- \`common_height\`: Desired height to which all images will be resized.
- \`output_path\`: Where the stitched image will be saved.

## Notes

- Ensure all images are in a format that is readable by the script.
- The evaluation of stitching quality is a simplification and is not a true stereo representation of the quality
- - **Important:** This image stitching approach is not as precise as methods used in stereo photogrammetry and might not be suitable for applications requiring high-precision measurements.

## Troubleshooting

- If images are not stitching correctly, ensure they are named sequentially and have sufficient overlap.
- Large differences in image sizes or non-overlapping images may cause the stitching to fail.

## Acknowledgements

- Special thanks to OpenCV 
