

import cv2


class GridGenerator:

    def __init__(self, pattern_size):

        self.pattern_size = pattern_size

    def generate(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        found, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size
        )

        output = image.copy()

        if found:

            cv2.drawChessboardCorners(
                output,
                self.pattern_size,
                corners,
                found
            )

        return output, found