# Table of Contents

- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Deployment](#deployment)

---

# 📌Project Overview

GridLens-Calib is a computer vision project developed to estimate camera calibration parameters and radial lens distortion from a single checkerboard image. The system performs checkerboard corner detection, estimates the camera intrinsic matrix, principal point, extrinsic parameters, and distortion coefficients, applies robust cost functions and RANSAC-based outlier filtering, generates an undistorted image and checkerboard grid, and evaluates calibration quality through reprojection error analysis.

The complete calibration pipeline is implemented using a modular architecture, enabling each stage of the workflow to be executed, analyzed, and validated independently while producing accurate and reproducible calibration results.

---

# ⚙️How It Works

| Step | Description |
|------|-------------|
| 1 | Load the checkerboard calibration image. |
| 2 | Detect checkerboard corners with sub-pixel refinement. |
| 3 | Estimate the camera intrinsic matrix, principal point, extrinsic parameters, and lens distortion coefficients. |
| 4 | Evaluate robust cost functions using Least Squares, Huber Loss, and Cauchy Loss. |
| 5 | Apply RANSAC to identify and remove outlier feature correspondences. |
| 6 | Prepare and refine camera calibration parameters through the optimization pipeline. |
| 7 | Undistort the input image using the estimated camera model. |
| 8 | Generate the undistorted checkerboard grid. |
| 9 | Reproject the estimated grid onto the original image and compute residuals and reprojection error. |
| 10 | Generate calibration statistics and export the final evaluation report. |

---

# ✨Features

- Automatic checkerboard corner detection
- Sub-pixel corner refinement
- Camera intrinsic parameter estimation
- Camera extrinsic parameter estimation
- Principal point estimation
- Radial and tangential distortion estimation
- Robust cost function evaluation
- RANSAC-based outlier removal
- Camera parameter optimization pipeline
- Image undistortion
- Undistorted checkerboard grid generation
- Reprojection residual visualization
- Reprojection error computation
- Calibration evaluation report generation
- Modular and maintainable Python implementation

---

# 💻Tech Stack

| Layer | Technology |
|--------|------------|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Numerical Computing | NumPy |
| Optimization | SciPy |
| Visualization | Matplotlib |
| Development Environment | Visual Studio Code |
| Version Control | Git & GitHub |

---

# 🚀Installation & Setup

### Clone Repository

```bash
git clone https://github.com/Naveenchandra29/GridLens-Calib.git
cd GridLens-Calib
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
cd src
python main.py
```

---

# 🌍Deployment

The project is designed as a standalone Python application and does not require any additional frontend or web framework. Running the calibration pipeline automatically processes the checkerboard image, estimates camera calibration parameters, computes distortion coefficients, performs outlier filtering and optimization, generates the undistorted image and grid, visualizes reprojection residuals, and exports the calibration summary for further evaluation.

The generated outputs are automatically stored in the **data/output/** and **results/** directories.

The project is named **GridLens-Calib**, where **Grid** represents the planar checkerboard grid used as the calibration pattern, **Lens** refers to the estimation and correction of camera lens distortion, and **Calib** represents the camera calibration process. The name reflects the complete workflow of extracting grid features to estimate accurate camera parameters and improve image geometry through calibration.
