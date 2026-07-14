from dataclasses import dataclass
import numpy as np


@dataclass
class CameraModel:
    """
    Stores all camera calibration parameters.
    """

    # -------- Intrinsic Parameters --------
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    # -------- Distortion Parameters --------
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0

    # -------- Extrinsic Parameters --------
    rotation_matrix: np.ndarray = None
    translation_vector: np.ndarray = None