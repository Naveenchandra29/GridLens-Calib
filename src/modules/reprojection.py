import cv2
import numpy as np


class Reprojection:

    @staticmethod
    def project_points(
        object_points,
        rotation_matrix,
        translation_vector,
        camera_matrix,
        distortion
    ):

        rvec, _ = cv2.Rodrigues(rotation_matrix)

        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            translation_vector,
            camera_matrix,
            distortion
        )

        return projected_points

    @staticmethod
    def compute_error(
        detected_points,
        projected_points
    ):

        detected = detected_points.reshape(-1, 2)

        projected = projected_points.reshape(-1, 2)

        error = np.linalg.norm(
            detected - projected,
            axis=1
        )

        mean_error = np.mean(error)

        rms_error = np.sqrt(
            np.mean(error ** 2)
        )

        max_error = np.max(error)

        return {
            "mean": mean_error,
            "rms": rms_error,
            "max": max_error
        }