from logistic_regression.logistic_regression import LogisticRegression
from deep_neural_network.deep_neural_network import DeepNeuralNetwork
from logistic_regression.data_preprocessor.cat_vs_non_cat_data_pre_processor import cat_vs_non_cat_data_pre_processor

logistic_regression = LogisticRegression(cat_vs_non_cat_data_pre_processor)
logistic_regression.build_classifier()
train_accuracy = logistic_regression.calculate_accuracy(x=logistic_regression.train_x, y=logistic_regression.train_y)
test_accuracy = logistic_regression.calculate_accuracy(x=logistic_regression.test_x, y=logistic_regression.test_y)

print(train_accuracy)
print(test_accuracy)


dnn = DeepNeuralNetwork(data_processor_callback=cat_vs_non_cat_data_pre_processor, layer_dims=[4, 1])
dnn.build_classifier()
train_accuracy = dnn.calculate_accuracy(x=dnn.train_x, y=dnn.train_y)
test_accuracy = dnn.calculate_accuracy(x=dnn.test_x, y=dnn.test_y)

print(train_accuracy)
print(test_accuracy)
