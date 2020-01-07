import numpy as np
from mixins.numpy_mixin import NumPyMixin
from enums.neural_network_activation_enum import NeuralNetworkActivationEnum


class NeuralNetworkMixin(NumPyMixin):
    def __init__(self, data_processor_callback):
        self.train_x, self.train_y, self.test_x, self.test_y = data_processor_callback()
        self.num_of_training_examples = self.train_x.shape[1]
        self.num_of_test_examples = self.test_x.shape[1]
        self.num_of_features = self.train_x.shape[0]
        self.initialize_weight_parameters()

    def predict(self, x):
        m = x.shape[1]
        y_prediction = np.zeros((1, m))
        a = self.forward_propagation(x)
        for i in range(a.shape[1]):
            y_prediction[0][i] = 1 if a[0][i] > .5 else 0
        return y_prediction

    def calculate_accuracy(self, x, y):
        y_prediction = self.predict(x)
        return 100 - np.mean(np.abs(y_prediction - y)) * 100

    @staticmethod
    def compute_cost(a, y):
        m = y.shape[1]
        cost_matrix = - (y * np.log(a) + (1 - y) * np.log(1 - a))
        cost = np.sum(cost_matrix, axis=1, keepdims=True) / m
        cost = np.squeeze(cost)
        return cost

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    def sigmoid_backward(self, da, z):
        a = self.sigmoid(z)
        dz = da * a * (1 - a)
        return dz

    @staticmethod
    def relu_backward(da, z):
        dz = np.array(da, copy=True)
        dz[z <= 0] = 0
        return dz

    @staticmethod
    def tanh_backward(da, z):
        a = np.tanh(z)
        dz = da * (1 - np.square(a))
        return dz

    def get_activation_function(self, activation):
        if activation == NeuralNetworkActivationEnum.Sigmoid:
            return self.sigmoid
        elif activation == NeuralNetworkActivationEnum.Relu:
            return self.relu
        elif activation == NeuralNetworkActivationEnum.Tanh:
            return np.tanh

    def get_backward_activation_function(self, activation):
        if activation == NeuralNetworkActivationEnum.Sigmoid:
            return self.sigmoid_backward
        elif activation == NeuralNetworkActivationEnum.Relu:
            return self.relu_backward
        elif activation == NeuralNetworkActivationEnum.Tanh:
            return self.tanh_backward

    def initialize_weight_parameters(self):
        raise NotImplementedError("You need to implement this method in your class")

    def forward_propagation(self, x):
        raise NotImplementedError("You need to implement this method in your class")

    def back_propagation(self, x, y, a):
        raise NotImplementedError("You need to implement this method in your class")

    def update_weight(self, dw, db, learning_rate):
        raise NotImplementedError("You need to implement this method in your class")

    def build_classifier(self, learning_rate=.05, number_of_iterations=2000):
        raise NotImplementedError("You need to implement this method in your class")
