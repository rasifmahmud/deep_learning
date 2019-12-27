import numpy as np


class LogisticRegression:
    def __init__(self, data_processor_callback):
        self.train_x, self.train_y, self.test_x, self.test_y = data_processor_callback()
        self.num_of_training_examples = self.train_x.shape[1]
        self.num_of_test_examples = self.test_x.shape[1]
        self.num_of_features = self.train_x.shape[0]
        self.weight_vector = self.get_empty_weight_vector()
        self.b = 0

    def get_empty_weight_vector(self):
        weight_vector = np.zeros((self.num_of_features, 1))
        return weight_vector

    @staticmethod
    def calculate_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, x):
        z = np.dot(self.weight_vector.transpose(), x) + self.b
        a = self.calculate_sigmoid(z)
        return a

    @staticmethod
    def back_propagation(x, y, a):
        m = x.shape[1]
        cost = np.sum(- (y * np.log(a) + (1 - y) * np.log(1 - a)) / m)
        dz = a - y
        dw = np.dot(x, dz.transpose()) / m
        db = np.sum(dz) / m
        return dw, db, cost

    def build_classifier(self, learning_rate=.05, number_of_iterations=2000):
        for i in range(number_of_iterations):
            a = self.forward_propagation(x=self.train_x)
            dw, db, cost = self.back_propagation(x=self.train_x, y=self.train_y, a=a)
            self.update_weight(dw, db, learning_rate)

    def update_weight(self, dw, db, learning_rate):
        self.weight_vector -= learning_rate * dw
        self.b -= learning_rate * db

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
