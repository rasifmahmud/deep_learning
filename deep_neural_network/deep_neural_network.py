from mixins.neural_metwork_mixin import NeuralNetworkMixin
from enums.neural_network_activation_enum import NeuralNetworkActivationEnum


# layer_dims = [3, 4,, 5, 1]

# layer_dict = [
#     {"activation": NeuralNetworkActivationEnum.Relu.value,
#     "weight_matrix": ...,
#      "b": ...,
#     },
#     {"activation": NeuralNetworkActivationEnum.Relu.value},
#     {"activation": NeuralNetworkActivationEnum.Sigmoid.value}
# ]
class DeepNeuralNetwork(NeuralNetworkMixin):
    def __init__(self, data_processor_callback, layer_dims, layer_dict_list=None, *args, **kwargs):
        super(DeepNeuralNetwork, self).__init__(data_processor_callback=data_processor_callback)
        self.layer_dims = layer_dims  # only keeps track of the number of neurons for different layers
        self.layer_dict_list = layer_dict_list  # keeps track of different configuration for different layers
        self.handle_layer_info()  # initializes layer_dict_list based on layer_dims in case layer_dict_list is None
        self.initialize_weight_parameters()

    def handle_layer_info(self):
        # if both of them are none
        if not self.layer_dims:
            raise Exception('You need to pass a list of indicating the number of neurons in each layer')
        if not self.layer_dict_list:
            self.layer_dict_list = [{"activation": NeuralNetworkActivationEnum.Relu.value} for i in
                                    range(len(self.layer_dims) - 1)]
            self.layer_dict_list.append({"activation": NeuralNetworkActivationEnum.Sigmoid.value})

    def initialize_weight_parameters(self):
        for index in range(self.layer_dims):
            if index:
                previous_layer_num_of_neurons = self.layer_dims[index - 1]
            else:
                previous_layer_num_of_neurons = self.num_of_features
            current_layer_num_of_neurons = self.layer_dims[index]
            weight_factor = .01
            self.layer_dict_list[index]['weight_matrix'] = self.get_random_matrix(current_layer_num_of_neurons,
                                                                                  previous_layer_num_of_neurons) * weight_factor
            self.layer_dict_list[index]['b'] = self.get_zero_matrix(current_layer_num_of_neurons, 1)
