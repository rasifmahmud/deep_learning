import numpy as np


class NumPyMixin:
    @staticmethod
    def get_zero_matrix(num_of_rows, num_of_colms):
        weight_vector = np.zeros((num_of_rows, num_of_colms))
        return weight_vector

    @staticmethod
    def get_random_matrix(num_of_rows, num_of_colms):
        np.random.seed(1)
        weight_vector = np.random.randn(num_of_rows, num_of_colms)
        return weight_vector
