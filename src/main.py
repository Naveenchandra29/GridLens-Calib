from modules.image_loader import ImageLoader
from modules.corner_detector import CornerDetector
import cv2
import cv2
from modules.camera_calibrator import CameraCalibrator
from modules.reprojection import Reprojection

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
    
if __name__ == "__main__":
    main()    