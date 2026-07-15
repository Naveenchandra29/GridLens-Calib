import cv2


class Undistorter:

    @staticmethod
    def undistort(image,
                  camera_matrix,
                  distortion):

        h, w = image.shape[:2]

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            (w, h),
            1,
            (w, h)
        )

        undistorted = cv2.undistort(
            image,
            camera_matrix,
            distortion,
            None,
            new_camera_matrix
        )

        return undistorted, roi