import cv2
import numpy as np

from modules.camera_model import CameraModel


class CameraCalibrator:

    def __init__(self, pattern_size, square_size=1.0):

        self.pattern_size = pattern_size
        self.square_size = square_size

    def generate_object_points(self):
        """
        Generate 3D checkerboard coordinates.
        """

        object_points = np.zeros(
            (self.pattern_size[0] * self.pattern_size[1], 3),
            np.float32
        )

        object_points[:, :2] = np.mgrid[
            0:self.pattern_size[0],
            0:self.pattern_size[1]
        ].T.reshape(-1, 2)

        object_points *= self.square_size

        return object_points

    def calibrate(self, image, corners):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        object_points = self.generate_object_points()

        objpoints = [object_points]
        imgpoints = [corners]

        success, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            None,
            None
        )

        if not success:
            raise RuntimeError("Camera calibration failed.")

        model = CameraModel()

        model.fx = camera_matrix[0, 0]
        model.fy = camera_matrix[1, 1]

        model.cx = camera_matrix[0, 2]
        model.cy = camera_matrix[1, 2]

        model.k1 = distortion[0][0]
        model.k2 = distortion[0][1]

        if distortion.shape[1] >= 5:
            model.k3 = distortion[0][4]

        model.rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
        model.translation_vector = tvecs[0]

        return model, camera_matrix, distortion