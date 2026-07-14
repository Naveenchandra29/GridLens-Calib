"""
===========================================================
GridLens-Calib
Module : Image Loader
Author : M Naveen Chandra
===========================================================

Purpose:
    Loads the calibration image and performs basic validation
    before it enters the camera calibration pipeline.
"""

from pathlib import Path
import cv2


class ImageLoader:
    """
    Handles image loading and validation.
    """

    def __init__(self, image_path):
        self.image_path = Path(image_path)

    def load(self):
        """
        Loads an image from disk.

        Returns
        -------
        image : numpy.ndarray
            Loaded image.
        """

        if not self.image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {self.image_path}"
            )

        image = cv2.imread(str(self.image_path))

        if image is None:
            raise ValueError(
                "OpenCV failed to read the image."
            )

        return image

    @staticmethod
    def print_info(image, image_path):

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        print("=" * 55)
        print("GridLens-Calib : Image Information")
        print("=" * 55)
        print(f"Image Name : {Path(image_path).name}")
        print(f"Resolution : {width} x {height}")
        print(f"Channels   : {channels}")
        print(f"Data Type  : {image.dtype}")
        print(f"Status     : Ready for Calibration")
        print("=" * 55)