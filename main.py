import numpy as np
import pandas as pds
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plot

#globals
classes = 10
epochs = 30
batch = 88
height = 28
width = 28


def load_train(path):
    train_csv = pds.read_csv(path)   #read the train data csv file
    #print(train_csv.head())

    # first column is the class label, and every following column has values for each of the 784 pixels in the image
    x = train_csv.iloc[:, 1:].values #pixel values in numpy array format
    y = train_csv.iloc[:, 0].values #labels also in numpy array format
    #print(type(x_train))

    print("TRAIN: The input shape is" + str(x.shape))
    print("TRAIN: The labels shape is" + str(y.shape))

    #convert vector to binary class matrix
    y = keras.utils.to_categorical(y)

    #split into train and validation subsets for better optimization
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 11)

    #reshape input train into images of single channel
    x_train = x_train.reshape(x_train.shape[0], height, width, 1)
    x_val = x_val.reshape(x_val.shape[0], height, width, 1)
    print("TRAIN: The train set is of shape: " + str(x_train.shape))
    print("TRAIN: The validation set is of shape: " + str(x_val.shape))

    #convert values to float32 type as required
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')

    #now normalize the pixels to a range of unity
    x_train /= 255
    x_val /= 255

    return x_train, x_val, y_train, y_val

def load_test(path):
    # load test data
    test_csv = pds.read_csv(path)

    x = test_csv.iloc[:, 1:].values
    y = test_csv.iloc[:, 0].values
    print("TEST: The input shape is" + str(x.shape))
    print("TEST: The labels shape is" + str(y.shape))

    y_test = keras.utils.to_categorical(y)

    x_test = x.reshape(x.shape[0], height, width, 1)

    x_test = x_test.astype('float32')

    x_test /= 255

    return x_test, y_test, test_csv

def conv_model(no_of_classes):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape = (28, 28, 1), activation = 'relu', padding = 'same')) #first convolution unit
    model.add(Conv2D(32, (4, 4), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides = 2))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))  # second convolution unit
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(Conv2D(64, (2, 2), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))  # third convolution unit
    
    model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation = 'relu')) #final dense layer for classification
    model.add(Dropout(0.3))
    model.add(Dense(no_of_classes, activation = 'softmax'))  #final output layer with 10 classes

    return model

def augment_data(model, x_train, x_test, y_train, y_test):
    # randomly augment data to reduce overfitting
    datagen = ImageDataGenerator(zoom_range = 0.2,
            rotation_range = 10,
            width_shift_range = 0.1,
            height_shift_range = 0.1,
            horizontal_flip = True,
            vertical_flip = False)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch),
                                  steps_per_epoch = int(np.ceil(x_train.shape[0] / float(batch_size))),   #borrowed from stack overflow answer - steps_per_epoch: Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch.
                                  #It should typically be equal to the number of unique samples of your dataset divided by the batch size.
                                  epochs = epochs,
                                  validation_data = (x_test, y_test),
                                  max_queue_size = 10,
                                  workers = 3, use_multiprocessing = True)

    evaluated = model.evaluate(x_test, y_test)

    return history, evaluated


def plot_accuracy(fit):
    plot.figure(figsize = [8, 8])
    plot.plot(fit.history['val_acc'],'g',linewidth = 2.0)
    plot.plot(fit.history['acc'],'r', linewidth = 2.0)
    plot.legend(['Training Accuracy', 'Validation Accuracy'], fontsize = 14)
    plot.xlabel('epochs ', fontsize = 14)
    plot.ylabel('accuracy', fontsize = 14)
    plot.title('accuracy vs epochs', fontsize = 14)
    plot.show()

def plot_loss(fit):
    plot.figure(figsize = [8, 8])
    plot.plot(fit.history['val_loss'],'g', linewidth = 2.0)
    plot.plot(fit.history['loss'],'r', linewidth = 2.0)
    plot.legend(['Training loss', 'Validation Loss'], fontsize = 14)
    plot.xlabel('epochs ', fontsize = 14)
    plot.ylabel('loss', fontsize = 14)
    plot.title('loss vs epochs', fontsize = 14)
    plot.show()

def prediction(conv, x_test, test):
    predicted = conv.predict_classes(x_test)

    #get all samples for plotting
    y_actual = test.iloc[:, 0]
    correct = np.nonzero(predicted == y_actual)[0]
    incorrect = np.nonzero(predicted != y_actual)[0]

    targets = ["Class {}".format(i) for i in range(classes)]
    print(classification_report(y_actual, predicted, target_names = targets))


if __name__ == '__main__':
    start = time.time()
    augment_flag = input("Enter 1 for augmentation, else enter 0: ")
    x_train, x_val, y_train, y_val = load_train('fashionmnist/fashion-mnist_train.csv')
    conv = conv_model(classes)

    conv.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    print(conv.summary()) #print the summary of the model

    x_test, y_test, test_csv = load_test('fashionmnist/fashion-mnist_test.csv') #load the test dataset

    if augment_flag != 1:
        history = conv.fit(x_train, y_train, batch_size = batch,
              epochs = epochs,
              verbose = 1,
              validation_data = (x_val, y_val))

        evaluated = conv.evaluate(x_test, y_test, verbose=0)

    else:
        history, evaluated = augment_data(conv, x_train, x_test, y_train, y_test)

    plot_accuracy(history) # plot the accuracy curves
    plot_loss(history) # plot the loss curves

    print('MAIN: Loss for Test set: ', evaluated[0])
    print('MAIN: Accuracy for Test set: ', evaluated[1])

    print("Full process took: " + str(time.time()-start) + " amount of time.")

    prediction(conv, x_test, test_csv) #final predicted metrics