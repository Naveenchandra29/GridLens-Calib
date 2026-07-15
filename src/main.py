from matplotlib import image

from modules.image_loader import ImageLoader
from modules.corner_detector import CornerDetector
import cv2
import cv2
from modules.camera_calibrator import CameraCalibrator
from modules.reprojection import Reprojection
from modules.cost_function import CostFunction
from modules.ransac_filter import RANSACFilter
from modules.optimizer import Optimizer
import numpy as np
from modules.undistorter import Undistorter
from modules.visualizer import Visualizer


print("OpenCV Version:", cv2.__version__)

def main():

    IMAGE_PATH = "../data/calibration_images/checkerboard.jpg"

    OUTPUT_PATH = "../data/output/detected_corners.jpg"

    PATTERN_SIZE = (9, 6)   # <-- We'll change this if needed

    loader = ImageLoader(IMAGE_PATH)

    image = loader.load()

    loader.print_info(image, IMAGE_PATH)

    print("\nRunning Corner Detection...")

    detector = CornerDetector(PATTERN_SIZE)

    gray, corners = detector.detect(image)

    result = detector.draw(
        image,
        PATTERN_SIZE,
        corners
    )

    detector.save(
        result,
        OUTPUT_PATH
    )

    print(f"Detected {len(corners)} corners.")

    print("Corner image saved successfully.")
    print("\nRunning Camera Calibration...")

    calibrator = CameraCalibrator(
        pattern_size=PATTERN_SIZE,
        square_size=1.0
    )

    camera, camera_matrix, distortion = calibrator.calibrate(
        image,
        corners
    )

    print("\nEstimated Camera Matrix\n")
    print(camera_matrix)

    print("\nEstimated Distortion\n")
    print(distortion)

    object_points = calibrator.generate_object_points()

    projected = Reprojection.project_points(
        object_points,
        camera.rotation_matrix,
        camera.translation_vector,
        camera_matrix,
        distortion
    )

    errors = Reprojection.compute_error(
        corners,
        projected
    )

    print("\nReprojection Error")
    print("----------------------")
    print(f"Mean : {errors['mean']:.4f} pixels")
    print(f"RMS  : {errors['rms']:.4f} pixels")
    print(f"Max  : {errors['max']:.4f} pixels")

    print("\nComputing Robust Cost Functions...")

    costs = CostFunction.summarize(
       corners,
       projected
    )

    print("\nCost Function Summary")
    print("---------------------------")

    for name, value in costs.items():
        print(f"{name:15s}: {value:.4f}")

    print("\nRunning RANSAC Outlier Removal...")

    ransac = RANSACFilter(threshold=1.5)

    (
        detected_inliers,
        projected_inliers,
        inlier_mask,
        residuals
    ) = ransac.filter(
        corners,
        projected
    )

    print(f"Total Points : {len(corners)}")
    print(f"Inliers      : {len(detected_inliers)}")
    print(f"Outliers     : {len(corners)-len(detected_inliers)}")    

    print("\nRunning Optimization Demo...")

    optimizer = Optimizer()

# Initial parameter
    initial = [5.0]

# Objective function
    def objective(x):
        return np.array([x[0] - 2])

    result = optimizer.optimize(
        initial,
        objective
    )

    optimizer.print_summary(result)

    print("Optimized Value :", result.x)
    print("\nUndistorting Image...")

    undistorted, roi = Undistorter.undistort(
        image,
        camera_matrix,
        distortion
    )

    cv2.imwrite(
        "../data/output/undistorted_checkerboard.jpg",
        undistorted
    )

    print("Undistorted image saved successfully.")
    print("\nGenerating Residual Visualization...")

    residual_image = Visualizer.draw_residuals(
        image,
        corners,
        projected
    )

    cv2.imwrite(
        "../data/output/residual_visualization.jpg",
        residual_image
    )

    print("Residual visualization saved.")

    
if __name__ == "__main__":
    main()    