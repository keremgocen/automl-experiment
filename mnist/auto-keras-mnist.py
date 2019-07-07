from keras.datasets import mnist
from autokeras.image.image_supervised import ImageClassifier, Constant
from autokeras.utils import pickle_to_file

import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

# Constant.MAX_ITER_NUM = 1
Constant.MAX_MODEL_NUM = 1
# Constant.MAX_NO_IMPROVEMENT_NUM = 1
Constant.SEARCH_MAX_ITER = 1


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train.reshape(x_train.shape + (1,))
    # x_test = x_test.reshape(x_test.shape + (1,))

    # clf = ImageClassifier(path='output/', verbose=True, searcher_args={
    #                       'trainer_args': {'max_iter_num': 1,
    #                                        'max_no_improvement_num': 1}})
    # clf.fit(x_train, y_train, time_limit=1 * 60 * 30)
    # clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    # y = clf.evaluate(x_test, y_test)
    # print(y)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    clf = ImageClassifier(verbose=True, searcher_args={'trainer_args':{'max_iter_num':7}})
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y)

    clf.export_autokeras_model('output/auto_mnist_model')

    # alternative
    best_model = clf.cnn.best_model.produce_model()
    pickle_to_file(best_model, 'output/auto_mnist_best_model')
    print(best_model)

# Step 2 : After the model training is complete, run examples/visualize.py, whilst passing the same path as parameter
# if __name__ == '__main__':
#    visualize('~/automodels/')
