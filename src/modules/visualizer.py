import cv2
import numpy as np


class Visualizer:

    @staticmethod
    def draw_residuals(
        image,
        detected_points,
        projected_points
    ):

        output = image.copy()

        detected = detected_points.reshape(-1, 2)

        projected = projected_points.reshape(-1, 2)

        for d, p in zip(detected, projected):

            d = tuple(np.int32(d))
            p = tuple(np.int32(p))

            # detected corner (green)
            cv2.circle(
                output,
                d,
                4,
                (0,255,0),
                -1
            )

            # projected corner (red)
            cv2.circle(
                output,
                p,
                3,
                (0,0,255),
                -1
            )

            # residual line
            cv2.line(
                output,
                d,
                p,
                (255,0,0),
                1
            )

        return output