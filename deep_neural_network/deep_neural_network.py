from mixins.neural_metwork_mixin import NeuralNetworkMixin
from enums.neural_network_activation_enum import NeuralNetworkActivationEnum
import numpy as np


# layer_dims = [3, 4,, 5, 1]

# layer_dict = [
#     {"activation": NeuralNetworkActivationEnum.Relu.value,
#     "w": ...,
#      "b": ...,
#     },
#     {"activation": NeuralNetworkActivationEnum.Relu.value},
#     {"activation": NeuralNetworkActivationEnum.Sigmoid.value}
# ]
class DeepNeuralNetwork(NeuralNetworkMixin):
    def __init__(self, data_processor_callback, layer_dims, layer_dict_list=None, *args, **kwargs):
        self.layer_dims = layer_dims  # only keeps track of the number of neurons for different layers
        self.layer_dict_list = layer_dict_list  # keeps track of different configuration for different layers
        self.handle_layer_info()  # initializes layer_dict_list based on layer_dims in case layer_dict_list is None
        super(DeepNeuralNetwork, self).__init__(data_processor_callback=data_processor_callback)

    def handle_layer_info(self):
        # if both of them are none
        if not self.layer_dims:
            raise Exception('You need to pass a list of indicating the number of neurons in each layer')
        if not self.layer_dict_list:
            self.layer_dict_list = [{"activation": NeuralNetworkActivationEnum.Relu} for i in
                                    range(len(self.layer_dims) - 1)]
            self.layer_dict_list.append({"activation": NeuralNetworkActivationEnum.Sigmoid})

    def initialize_weight_parameters(self):
        for index, dimension in enumerate(self.layer_dims):
            if index:
                previous_layer_num_of_neurons = self.layer_dims[index - 1]
            else:
                previous_layer_num_of_neurons = self.num_of_features
            current_layer_num_of_neurons = dimension
            weight_factor = .01
            self.layer_dict_list[index]['w'] = self.get_random_matrix(current_layer_num_of_neurons,
                                                                      previous_layer_num_of_neurons) * weight_factor
            self.layer_dict_list[index]['b'] = self.get_zero_matrix(current_layer_num_of_neurons, 1)

    def compute_cost(self, a):
        m = self.num_of_training_examples
        Y = self.train_y
        cost_matrix = - (Y * np.log(a) + (1 - Y) * np.log(1 - a))
        cost = np.sum(cost_matrix, axis=1, keepdims=True) / m
        cost = np.squeeze(cost)
        return cost

    def forward_propagation(self, x):
        a = x
        for layer_dict in self.layer_dict_list:
            layer_dict['a_prev'] = a
            z = np.dot(layer_dict['w'], a) + layer_dict['b']
            activation_function = self.get_activation_function(layer_dict['activation'])
            a = activation_function(z)
            layer_dict['z'] = z
            layer_dict['a'] = a

        return a

    def linear_backward(self, dz, layer_dict):
        a_prev = layer_dict['a_prev']
        w = layer_dict['w']
        m = self.num_of_training_examples

        dw = np.dot(dz, a_prev.transpose()) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        da_prev = np.dot(w.transpose(), dz)

        return da_prev, dw, db

    def linear_activation_backward(self, da, layer_dict):
        backward_activation_function = self.get_backward_activation_function(layer_dict['activation'])

        dz = backward_activation_function(da, layer_dict['z'])
        da_prev, dw, db = self.linear_backward(dz, layer_dict)

        layer_dict['da_prev'] = da_prev
        layer_dict['dw'] = dw
        layer_dict['db'] = db

        return da_prev

    def back_propagation(self, x, y, a):
        da = - (np.divide(y, a) - np.divide(1 - y, 1 - a))
        for layer_dict in reversed(self.layer_dict_list):
            da = self.linear_activation_backward(da, layer_dict)

    def update_weight(self, learning_rate):
        for layer_dict in self.layer_dict_list:
            layer_dict['w'] -= learning_rate * layer_dict['dw']
            layer_dict['b'] -= learning_rate * layer_dict['db']

    def build_classifier(self, learning_rate=.05, number_of_iterations=2000):
        self.initialize_weight_parameters()
        for iteration in range(number_of_iterations):
            x = self.train_x
            y = self.train_y

            a = self.forward_propagation(self.train_x)
            cost = self.compute_cost(a)
            if iteration % 100 == 0:
                print(cost)
            self.back_propagation(x, y, a)
            self.update_weight(learning_rate=learning_rate)
