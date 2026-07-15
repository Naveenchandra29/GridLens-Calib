import numpy as np
from scipy.optimize import least_squares


class Optimizer:

    def __init__(self):

        self.loss = "huber"

        self.method = "trf"

    def optimize(
        self,
        initial_parameters,
        objective_function
    ):

        result = least_squares(
            objective_function,
            initial_parameters,
            method=self.method,
            loss=self.loss,
            verbose=2
        )

        return result

    @staticmethod
    def print_summary(result):

        print("\nOptimization Summary")
        print("--------------------------------")

        print("Success :", result.success)

        print("Message :", result.message)

        print("Cost :", result.cost)

        print("Iterations :", result.nfev)