import numpy as np


class NumPyMixin:
    @staticmethod
    def get_zero_matrix(num_of_rows, num_of_colms):
        weight_vector = np.zeros((num_of_rows, num_of_colms))
        return weight_vector

    @staticmethod
    def get_random_matrix(num_of_rows, num_of_colms):
        weight_vector = np.random.randn(num_of_rows, num_of_colms)
        return weight_vector

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)
