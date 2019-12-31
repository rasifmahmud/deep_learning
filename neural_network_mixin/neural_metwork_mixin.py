import numpy as np
from utils.numpy_mixin import NumPyMixin


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
