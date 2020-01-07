import h5py
import numpy as np


def cat_vs_non_cat_data_pre_processor(train_filename='./logistic_regression/dataset/train_catvnoncat.h5',
                                      test_filename='./logistic_regression/dataset/test_catvnoncat.h5'):
    train_file = h5py.File(train_filename, 'r')
    test_file = h5py.File(test_filename, 'r')

    train_x = np.array(train_file['train_set_x']).transpose()
    reshaped_train_x = train_x.reshape((train_x.shape[0] * train_x.shape[1] * train_x.shape[2], train_x.shape[3]))

    train_y = np.array(train_file['train_set_y'])
    reshaped_train_y = train_y.reshape((1, train_y.shape[0]))

    test_x = np.array(test_file['test_set_x']).transpose()
    reshaped_test_x = test_x.reshape((test_x.shape[0] * test_x.shape[1] * test_x.shape[2], test_x.shape[3]))

    test_y = np.array(test_file['test_set_y'])
    reshaped_test_y = test_y.reshape((1, test_y.shape[0]))

    return reshaped_train_x / 255, reshaped_train_y / 255, reshaped_test_x / 255, reshaped_test_y / 255
