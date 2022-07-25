# Step 1: Load in the data from data.json
# Step 2: Split the data into training and testing sets
# Step 3: Build the network architecture
# Step 4: Train the network

import json
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import LeakyReLU
DATA_PATH = "data.json"

def plot_history(history):
    figure, axis = plt.subplots(2)

    # Make the accuracy subplot
    axis[0].plot(history.history["accuracy"], label="train accuracy")
    axis[0].plot(history.history["val_accuracy"], label="test accuracy")
    axis[0].set_ylabel("Accuracy")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Accuracy eval")

    # Make the error subplot
    axis[1].plot(history.history["loss"], label="train error")
    axis[1].plot(history.history["val_loss"], label="test error")
    axis[1].set_ylabel("Error")
    axis[1].set_xlabel("Epoch")
    axis[1].legend(loc="upper right")
    axis[1].set_title("Error eval")

    plt.show()

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # Convert the list into a numpy array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

def music_classifier():
         # Load the data
        inputs, targets = load_data(DATA_PATH)

        # Give 30% of the data to the test set and the rest to the training set
        x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.3)
        # Construct the network achitecture
        model = keras.Sequential([
            # input layer
            keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
            # Hidden layers 1 through 3
            keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
            # keras.layers.LeakyReLU(alpha=0.05),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.003)),
            # keras.layers.LeakyReLU(alpha=0.05),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
            # keras.layers.LeakyReLU(alpha=0.05),
            keras.layers.Dropout(0.3),
            # Output Layer
            keras.layers.Dense(32, activation="relu"),
            # keras.layers.LeakyReLU(alpha=0.05),
            keras.layers.Dense(32, activation="relu"),
            # keras.layers.LeakyReLU(alpha=0.05),
            keras.layers.Dense(10, activation="softmax")
        ])

        # Compile the network
        # optimizer = keras.optimizers.Adam(learning_rate=0.01)
        optimizer = keras.optimizers.SGD(learning_rate=0.01)
        model.compile(optimizer, 
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.summary()

        # Train the network
        history = model.fit(x_train, 
                  y_train,
                  validation_data=(x_test, y_test),
                  epochs = 100,
                  batch_size=32)



        # Print  
        train_error, train_accuracy = model.evaluate(x_train, y_train, verbose=1)
        test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
        print("Accuracy on test set is: {}".format(test_accuracy))
        print("Error on test set is: {}".format(test_error))
        print("Accuracy on train set is: {}".format(train_accuracy))
        print("Error on train set is: {}".format(train_error))
        #plot accuracy and error over epochs
        plot_history(history)
     

def prepare_dataset(test_size, validation_size):
    # load data 
    x, y = load_data(DATA_PATH)
    # create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size)
    # make the train/validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)
    # 3d array -> (130, 13, 1) Third dimension is the channel/depth
    x_train = x_train[..., np.newaxis] # x_train turn into 4d array which includes number of samples
    x_validation = x_validation[...,np.newaxis]
    x_test = x_test[...,np.newaxis]

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def prepare_rnn_dataset(test_size, validation_size):
    # load data 
    x, y = load_data(DATA_PATH)
    # create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size)
    # make the train/validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)
    # 3d array -> (130, 13, 1) Third dimension is the channel/depth

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def build_rnn_model(input_shape):
    model = keras.Sequential()

    # Sequence to sequence layer
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    # Sequence to vector layer
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.LeakyReLU(0.05))
    model.add(keras.layers.Dropout(0.3))

    # Output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

def build_cnn_model(input_shape):
    # Make the model
    model = keras.Sequential()
    # layer 1: convolution/pooling/batchnorm
    model.add(keras.layers.Conv2D(32, (3,3), input_shape=input_shape, activation="tanh"))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # layer 2 = convolution/pooling/batchnorm
    model.add(keras.layers.Conv2D(32, (3,3), input_shape=input_shape, activation="tanh"))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # layer 3 = convolution/pooling/batchnorm
    model.add(keras.layers.Conv2D(32, (2,2), input_shape=input_shape, activation="tanh"))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    # # layer 3 = convolution/pooling/batchnorm
    # model.add(keras.layers.Conv2D(16, (1, 1), activation = "relu"))
    # model.add(keras.layers.MaxPool2D((1, 1), strides=(2, 2), padding="same"))
    # model.add(keras.layers.BatchNormalization())
    # Flatten the output
    model.add(keras.layers.Flatten())
    # layer 4 = Dense layer
    model.add(keras.layers.Dense(64, activation="tanh"))
    # model.add(LeakyReLU(alpha=0.05))
    model.add(keras.layers.Dropout(0.3)) # Prevent overfitting
    # Layer 5 = Output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def predict(model, x, y):
    # x must be in 4 dimensions
    x = x[np.newaxis, ...]
    prediction = model.predict(x) 

    # extract index with max calue
    predicted_index = np.argmax(prediction, axis=1)
    print("EXPECTED INDEX: {}, PREDICTED INDEX: {}".format(y, predicted_index))


def cnn_music_classifier():
    # Make a train validation and test sets
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_dataset(0.25, 0.2)
    # Built the CNN network
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    model = build_cnn_model(input_shape)
    # Compile the network
    optimizer =keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = keras.optimizers.SGD(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics = 'accuracy')
    # Train the CNN
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=40)
    # Test the CNN
    train_error, train_accuracy = model.evaluate(x_train, y_train, verbose=1)
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    print("Error on test set is: {}".format(test_error))
    print("Accuracy on train set is: {}".format(train_accuracy))
    print("Error on train set is: {}".format(train_error))
    # Make a prediction on a sample

    x = x_test[33]
    y = y_test[33]
    predict(model, x, y)
    model.summary()
    plot_history(history)

def rnn_music_classifier():
     # Make a train validation and test sets
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_rnn_dataset(0.25, 0.2)
    # Built the RNN network
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_rnn_model(input_shape)
    optimizer =keras.optimizers.Adam(learning_rate=0.001)
    # optimizer = keras.optimizers.SGD(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics = 'accuracy')
    
    # Train the RNN
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=40)
    # Test the RNN
    train_error, train_accuracy = model.evaluate(x_train, y_train, verbose=1)
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy on test set is: {}".format(test_accuracy))
    print("Error on test set is: {}".format(test_error))
    print("Accuracy on train set is: {}".format(train_accuracy))
    print("Error on train set is: {}".format(train_error))
    # Make a prediction on a sample

    x = x_test[33]
    y = y_test[33]
    predict(model, x, y)
    model.summary()
    plot_history(history)


if __name__ == "__main__":
    # music_classifier()
    # cnn_music_classifier()
    rnn_music_classifier()