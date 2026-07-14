from modules.image_loader import ImageLoader
from modules.corner_detector import CornerDetector
import cv2
import cv2

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

if __name__ == "__main__":
    main()    