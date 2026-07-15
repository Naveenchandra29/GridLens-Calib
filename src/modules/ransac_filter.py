import numpy as np


class RANSACFilter:

    def __init__(self, threshold=1.5):

        self.threshold = threshold

    def filter(self,
               detected_points,
               projected_points):

        detected = detected_points.reshape(-1, 2)

        projected = projected_points.reshape(-1, 2)

        residuals = np.linalg.norm(
            detected - projected,
            axis=1
        )

        inlier_mask = residuals < self.threshold

        detected_inliers = detected[inlier_mask]
        projected_inliers = projected[inlier_mask]

        return (
            detected_inliers,
            projected_inliers,
            inlier_mask,
            residuals
        )