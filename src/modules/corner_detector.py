from pathlib import Path
import cv2
import numpy as np


class CornerDetector:

    def __init__(self, pattern_size):

        self.pattern_size = pattern_size

        self.criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

    def detect(self, image):


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print(f"Pattern Size : {self.pattern_size}")
        print(f"Gray Shape   : {gray.shape}")

        flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE
        )

    # First try the standard detector
        found, corners = cv2.findChessboardCorners(
        gray,
        self.pattern_size,
        flags
        )

    # If it fails, try the more robust SB detector
        if not found:
            print("Standard detector failed. Trying SB detector...")

        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray,
                self.pattern_size,
                flags
            )

        if not found:
            raise ValueError(
            f"Checkerboard ({self.pattern_size}) could not be detected."
        )

    # cornerSubPix is only needed for the classic detector
        if corners.dtype != np.float32:
            corners = corners.astype(np.float32)

        refined_corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        self.criteria
        )

        return gray, refined_corners

    @staticmethod
    def draw(image, pattern_size, corners):

        output = image.copy()

        cv2.drawChessboardCorners(
            output,
            pattern_size,
            corners,
            True
        )

        return output

    @staticmethod
    def save(image, save_path):

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(str(save_path), image)