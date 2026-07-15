import numpy as np
from scipy.optimize import least_squares


class Optimizer:

    def __init__(self):

        self.method = "trf"
        self.loss = "huber"

    def optimize(self,
                 initial_parameters,
                 objective_function):

        result = least_squares(
            objective_function,
            initial_parameters,
            method=self.method,
            loss=self.loss,
            verbose=1
        )

        return result

    @staticmethod
    def print_summary(result):

        print("\nOptimization Summary")
        print("-----------------------------")
        print("Success :", result.success)
        print("Status  :", result.status)
        print("Cost    :", result.cost)
        print("Iterations :", result.nfev)