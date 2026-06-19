# Image Processing Workflow

This document details the image processing pipeline used for lane detection in the `lane_detector.py` module.

## 1. Preprocessing (`preprocess_frame`)
The goal of this step is to filter out noise and isolate the lane markers using color and gradient thresholding.
- **Blurring**: The RGB image is smoothed using Gaussian and median blur to reduce noise.
- **Color Filtering (LAB Space)**: 
  - The image is converted to LAB color space.
  - Green pixels (e.g., grass/vegetation) are masked out.
  - Red/warm pixels are retained.
- **Edge and Gradient Detection**: 
  - Converted to grayscale.
  - Sobel operators compute gradients in the x and y directions.
  - Gradient magnitude and direction are thresholded to isolate distinct lane edges.
- **White Line Detection**: Grayscale thresholding highlights bright white lane markers.
- **Combination and Cleaning**: All binary masks are combined. Small contour areas are filtered out to remove isolated noise pixels.

## 2. Perspective Transformation (`perspective_transform`)
The camera's perspective view is transformed into a top-down "bird's-eye view".
- Defines a Region of Interest (ROI) polygon in the original frame.
- Uses `cv2.getPerspectiveTransform` to map the ROI to a rectangular top-down space.
- `cv2.warpPerspective` applies this transformation to the binary image, making lane lines appear parallel and easing the curve fitting process.

## 3. Lane Center Line Extraction (`find_center_line`)
Identifies the exact pixels belonging to the lane line in the warped image.
- A sliding window approach is used.
- A histogram of the bottom half of the image identifies the initial base x-coordinate where the line is most concentrated.
- Vertically stacked windows slide upward, adjusting their x-center based on the mean x-position of non-zero pixels within them.
- Returns the x and y coordinates of all detected lane pixels.

## 4. Parameter Computation (`compute_lane_params`)
Computes the steering and position parameters from the detected pixels.
- Fits a 1st-degree polynomial (a straight line: `x = m*y + b`) to the pixel coordinates using `np.polyfit`.
- **Steering Angle (`theta`)**: Computed using the arctangent of the slope (`m`).
- **Lateral Offset (`b`)**: Calculated relative to the center of the image width to determine how far off-center the rover is.

## 5. Visualization (`plot_lane_lines`)
For debugging and monitoring.
- Generates a visual overlay of the detected lane line on the bird's-eye view.
- Uses the inverse perspective matrix (`M_inv`) to project the detected line back onto the original camera frame.
- Combines both views into a single diagnostic image.
