# https://www.kaggle.com/kernels/scriptcontent/3063488/download

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for plotting
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
import seaborn as sns
from subprocess import check_output
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

print(check_output(["ls", "./input"]).decode("utf8"))

# loading the dataset.......(Train)
train = pd.read_csv("./input/train.csv")
print(train.shape)
train.head()

z_train = Counter(train['label'])
z_train

sns.countplot(train['label'])

# loading the dataset.......(Test)
test = pd.read_csv("./input/test.csv")
print(test.shape)
test.head()

x_train = (train.ix[:, 1:].values).astype('float32')  # all pixel values
y_train = train.ix[:, 0].values.astype(
    'int32')  # only labels i.e targets digits
x_test = test.values.astype('float32')

# %matplotlib inline
# preview the images first
# plt.figure(figsize=(12, 10))
# x, y = 10, 4
# for i in range(40):
#     plt.subplot(y, x, i+1)
#     plt.imshow(x_train[i].reshape((28, 28)), interpolation='nearest')
# plt.show()

x_train = x_train/255.0
x_test = x_test/255.0


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

X_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
batch_size = 64

num_classes = 10
epochs = 20
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices One Hot Encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)

# keras


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_normal', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 kernel_initializer='he_normal'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.20))
model.add(Conv2D(64, (3, 3), activation='relu',
                 padding='same', kernel_initializer='he_normal'))
model.add(Conv2D(64, (3, 3), activation='relu',
                 padding='same', kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',
                 padding='same', kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0001)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=15,
    zoom_range=0.1,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

model.summary()

datagen.fit(X_train)
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        epochs=epochs, validation_data=(
                            X_val, Y_val),  # batch_size
                        verbose=1, steps_per_epoch=X_train.shape[0],
                        callbacks=[learning_rate_reduction],)
