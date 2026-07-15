import numpy as np


class RANSACFilter:

    def __init__(
        self,
        threshold=2.0,
        iterations=200,
        sample_size=8
    ):

        self.threshold = threshold
        self.iterations = iterations
        self.sample_size = sample_size

    def filter(
        self,
        detected_points,
        projected_points
    ):

        detected = detected_points.reshape(-1, 2)

        projected = projected_points.reshape(-1, 2)

        n = len(detected)

        best_mask = np.zeros(n, dtype=bool)

        best_inliers = 0

        rng = np.random.default_rng(42)

        for _ in range(self.iterations):

            sample = rng.choice(
                n,
                self.sample_size,
                replace=False
            )

            sample_error = np.mean(
                np.linalg.norm(
                    detected[sample] -
                    projected[sample],
                    axis=1
                )
            )

            residuals = np.linalg.norm(
                detected -
                projected,
                axis=1
            )

            mask = residuals < (
                sample_error +
                self.threshold
            )

            inliers = np.sum(mask)

            if inliers > best_inliers:

                best_inliers = inliers

                best_mask = mask

        return (
            detected[best_mask],
            projected[best_mask],
            best_mask,
            residuals
        )