import numpy as np
from mixins.neural_metwork_mixin import NeuralNetworkMixin


class LogisticRegression(NeuralNetworkMixin):
    def __init__(self, data_processor_callback):
        super(LogisticRegression, self).__init__(data_processor_callback=data_processor_callback)

    def initialize_weight_parameters(self):
        self.weight_vector = self.get_zero_matrix(self.num_of_features, 1)
        self.b = 0

    def forward_propagation(self, x):
        z = np.dot(self.weight_vector.transpose(), x) + self.b
        a = self.calculate_sigmoid(z)
        return a

    def back_propagation(self, x, y, a):
        m = x.shape[1]
        cost = np.sum(- (y * np.log(a) + (1 - y) * np.log(1 - a)) / m)
        dz = a - y
        dw = np.dot(x, dz.transpose()) / m
        db = np.sum(dz) / m
        return dw, db, cost

    def update_weight(self, dw, db, learning_rate):
        self.weight_vector -= learning_rate * dw
        self.b -= learning_rate * db

    def build_classifier(self, learning_rate=.05, number_of_iterations=2000):
        for i in range(number_of_iterations):
            a = self.forward_propagation(x=self.train_x)
            dw, db, cost = self.back_propagation(x=self.train_x, y=self.train_y, a=a)
            self.update_weight(dw, db, learning_rate)
