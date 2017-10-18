"""
@author Alexander Hamme

With inspiration and code from Dr. Jason Brownlee's Machine Learning blogs
"""
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend
from keras import models
from keras import layers
import numpy

NUMB_EPOCHS = 10
BATCH_SIZE = 200
RANDOM_SEED = 7

backend.set_image_dim_ordering('th')
numpy.random.seed(RANDOM_SEED)      # set random seed to enable easy reproducibility

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshape to be [samples][channels][width][height]
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = x_train / 255
X_test = x_test / 255

# perform one hot encoding on outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


def create_model():
    # create the model, add layers
    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(layers.Conv2D(15, (3, 3), activation='relu'))
    cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(layers.Dropout(0.2))
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(128, activation='relu'))
    cnn_model.add(layers.Dense(50, activation='relu'))
    cnn_model.add(layers.Dense(num_classes, activation='softmax'))
    # Compile the model
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return cnn_model

# build the model
model = create_model()
# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=NUMB_EPOCHS, batch_size=BATCH_SIZE)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN Error: {:2f}%".format(100 - scores[1] * 100))
