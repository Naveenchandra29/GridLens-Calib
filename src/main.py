"""
===========================================================
GridLens-Calib
Main Pipeline

Author : M Naveen Chandra

Description:
Main entry point for the camera calibration pipeline.
===========================================================
"""

from modules.image_loader import ImageLoader


def main():

    # --------------------------------------------------
    # Step 1 : Load Calibration Image
    # --------------------------------------------------

    IMAGE_PATH = "../data/calibration_images/checkerboard.jpg"

    loader = ImageLoader(IMAGE_PATH)

    image = loader.load()

    loader.print_info(image, IMAGE_PATH)


if __name__ == "__main__":
    main()