import numpy as np


class CostFunction:
    """
    Computes reprojection residuals and different
    robust loss functions.
    """

    @staticmethod
    def compute_residuals(detected_points, projected_points):
        """
        Compute reprojection residuals.

        Parameters
        ----------
        detected_points : ndarray
            Corner points detected in the image.

        projected_points : ndarray
            Corner points projected using the camera model.

        Returns
        -------
        residuals : ndarray
        """

        detected = detected_points.reshape(-1, 2)
        projected = projected_points.reshape(-1, 2)

        residuals = detected - projected

        return residuals

    @staticmethod
    def least_squares(residuals):
        """
        Standard Least Squares Cost
        """

        return np.sum(residuals ** 2)

    @staticmethod
    def huber_loss(residuals, delta=1.0):
        """
        Robust Huber Loss
        """

        absolute = np.abs(residuals)

        quadratic = np.minimum(absolute, delta)

        linear = absolute - quadratic

        return np.sum(
            0.5 * quadratic ** 2 +
            delta * linear
        )

    @staticmethod
    def cauchy_loss(residuals, c=2.3849):
        """
        Robust Cauchy Loss
        """

        return np.sum(
            c ** 2 *
            np.log1p(
                (residuals / c) ** 2
            )
        )

    @staticmethod
    def summarize(detected_points, projected_points):

        residuals = CostFunction.compute_residuals(
            detected_points,
            projected_points
        )

        return {
            "Least Squares":
                CostFunction.least_squares(residuals),

            "Huber":
                CostFunction.huber_loss(residuals),

            "Cauchy":
                CostFunction.cauchy_loss(residuals)
        }