from pathlib import Path


class Evaluation:

    @staticmethod
    def print_summary(
            image_name,
            image_shape,
            corners,
            inliers,
            errors,
            costs):

        print("\n")
        print("=" * 55)
        print("GridLens-Calib Final Evaluation")
        print("=" * 55)

        print(f"Image              : {image_name}")
        print(f"Resolution         : {image_shape[1]} x {image_shape[0]}")
        print(f"Detected Corners   : {corners}")
        print(f"Inliers            : {inliers}")
        print(f"Outliers           : {corners-inliers}")

        print("\nReprojection Error")
        print("------------------------------")
        print(f"Mean Error         : {errors['mean']:.4f} px")
        print(f"RMS Error          : {errors['rms']:.4f} px")
        print(f"Maximum Error      : {errors['max']:.4f} px")

        print("\nRobust Cost Functions")
        print("------------------------------")
        for key, value in costs.items():
            print(f"{key:18s}: {value:.4f}")

        print("\nCalibration Status : SUCCESS")
        print("=" * 55)

    @staticmethod
    def save_summary(
            output_path,
            image_name,
            image_shape,
            corners,
            inliers,
            errors,
            costs,
            camera_matrix,
            distortion):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_path, "w") as file:

            file.write("=" * 60 + "\n")
            file.write("GridLens-Calib Calibration Report\n")
            file.write("=" * 60 + "\n\n")

            file.write(f"Image : {image_name}\n")
            file.write(
                f"Resolution : {image_shape[1]} x {image_shape[0]}\n\n"
            )

            file.write(f"Detected Corners : {corners}\n")
            file.write(f"Inliers : {inliers}\n")
            file.write(f"Outliers : {corners-inliers}\n\n")

            file.write("Camera Matrix\n")
            file.write(str(camera_matrix))
            file.write("\n\n")

            file.write("Distortion Coefficients\n")
            file.write(str(distortion))
            file.write("\n\n")

            file.write("Reprojection Error\n")
            file.write(
                f"Mean : {errors['mean']:.4f}\n"
            )
            file.write(
                f"RMS : {errors['rms']:.4f}\n"
            )
            file.write(
                f"Maximum : {errors['max']:.4f}\n\n"
            )

            file.write("Cost Functions\n")

            for key, value in costs.items():
                file.write(f"{key}: {value:.4f}\n")

            file.write("\nCalibration Successful\n")