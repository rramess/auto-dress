from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

row, col = 28, 28 #Store size of data
shape = (row, col, 1) #Define shape
augment = 0
classes = 10
epochs = 1
batch_size = 10

def load_train (path):
    train = pd.read_csv('fashionmnist/fashion-mnist_train.csv') #Load training data

    data = np.array(train.iloc[:, 1:]) #Stores all non-label data
    labels = to_categorical(np.array(train.iloc[:, 0])) #Stores all labels

    # print(data.shape)
    # print(labels.shape)

    train_data, valid_data, train_label, valid_label = train_test_split(data, labels, test_size=0.2, random_state=15)

    # print(train_data.shape)
    # print(valid_data.shape)

    train_data = train_data.reshape(train_data.shape[0], row, col, 1)
    valid_data = valid_data.reshape(valid_data.shape[0], row, col, 1)

    train_data = train_data.astype('float32')
    valid_data = valid_data.astype('float32')

    train_data /= 255
    valid_data /= 255
    

    return train_data, train_label, valid_data, valid_label

def load_test (path):
    test = pd.read_csv('fashionmnist/fashion-mnist_test.csv') #Load test data

    data_test = np.array(test.iloc[:, 1:])
    labels_test = to_categorical(np.array(test.iloc[:, 0]))

    data_test = data_test.reshape(data_test.shape[0], row, col, 1)
    data_test = data_test.astype('float32')

    data_test /= 255

    return data_test, labels_test, test

def accuracy_plot (history):
    plot.figure(figsize = [8, 8])
    plot.plot(history['val_acc'],'g',linewidth = 2.0)
    plot.plot(history['acc'],'r', linewidth = 2.0)
    plot.legend(['Training Accuracy', 'Validation Accuracy'], fontsize = 14)
    plot.xlabel('epochs ', fontsize = 14)
    plot.ylabel('accuracy', fontsize = 14)
    plot.title('accuracy vs epochs', fontsize = 14)
    plot.show()

def loss_plot (history):
    plot.figure(figsize = [8, 8])
    plot.plot(history['val_loss'],'g', linewidth = 2.0)
    plot.plot(history['loss'],'r', linewidth = 2.0)
    plot.legend(['Training loss', 'Validation Loss'], fontsize = 14)
    plot.xlabel('epochs ', fontsize = 14)
    plot.ylabel('loss', fontsize = 14)
    plot.title('loss vs epochs', fontsize = 14)
    plot.show()

def class_report (test, data_test):
    predicted_classes = model.predict_classes(data_test)
    y_true = test.iloc[:, 0]
    correct = np.nonzero(predicted_classes==y_true)[0]
    incorrect = np.nonzero(predicted_classes!=y_true)[0]
    target_names = ["Class {}".format(i) for i in range(10)]
    print(classification_report(y_true, predicted_classes, target_names=target_names))




def Model_init (train_data, train_label, valid_data, valid_label, data_test, labels_test): 
    model = Sequential()

    model.add(Conv2D(32, (3, 3), strides = 2, input_shape=(28,28,1), activation='relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # second convolutional + max pooling layer
    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    # Third convolutional layer
    model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
    model.add(Dropout(0.4))

    model.add(Flatten())  
    model.add(Dense(128, activation='relu'))     
    model.add(Dense(10, activation='softmax'))  

    # Compiling CNN
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    model.summary()

    if augment == 0:
        history = model.fit(train_data, train_label, validation_data = (valid_data, valid_label), batch_size = batch_size, epochs = epochs)

    else:

        datagen = ImageDataGenerator(
#         zoom_range=0.2, # randomly zoom into images
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

        print(train_data.shape)
        history = model.fit_generator(datagen.flow(train_data, train_label, batch_size=batch_size),
                              epochs=epochs,
                              validation_data=(valid_data, valid_label),
                              workers=4)


    scores = model.evaluate(data_test, labels_test, verbose = 1)
    print ('Accuracy : {}'.format(scores[1]))


    return model, history




if __name__ == "__main__":

    
    train_data, train_label, valid_data, valid_label = load_train('fashionmnist/fashion-mnist_train.csv')
    #print(train_data.shape)
    data_test, labels_test, test = load_test('fashionmnist/fashion-mnist_test.csv')

    # ***Preprocessing done ***

    conv_model, history = Model_init(train_data, train_label, valid_data, valid_label, data_test, labels_test)
    loss_plot (history)
    accuracy_plot (history)
    class_report (test, data_test)

    
    