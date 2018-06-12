from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def createModel():
    
    batch_size = 200
    classes = 10
    epochs = 5
    row, col = 28, 28
    
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

    return model




if __name__ == "__main__":

    train = pd.read_csv('input/fashion-mnist_train.csv') #Load training data
    test = pd.read_csv('input/fashion-mnist_test.csv') #Load test data

    row, col = 28, 28 #Store size of data
    shape = (row, col, 1) #Define shape

    data = np.array(train.iloc[:, 1:]) #Stores all non-label data
    labels = to_categorical(np.array(train.iloc[:, 0])) #Stores all labels

    print(data.shape)
    print(labels.shape)

    train_data, valid_data, train_label, valid_label = train_test_split(data, labels, test_size=0.2, random_state=15)

    print(train_data.shape)
    print(valid_data.shape)

    data_test = np.array(test.iloc[:, 1:])
    labels_test = to_categorical(np.array(test.iloc[:, 0]))

    train_data = train_data.reshape(train_data.shape[0], row, col, 1)
    data_test = data_test.reshape(data_test.shape[0], row, col, 1)
    valid_data = valid_data.reshape(valid_data.shape[0], row, col, 1)

    train_data = train_data.astype('float32')
    data_test = data_test.astype('float32')
    valid_data = valid_data.astype('float32')

    train_data /= 255
    valid_data /= 255
    data_test /= 255

    # ***Preprocessing done ***

    model = createModel()
    history = model.fit(train_data, train_label, validation_data = (valid_data, valid_label), batch_size = 100, epochs = 50)

    scores = model.evaluate(valid_data, valid_label, verbose = 1)
    print ('Accuracy : {}'.format(scores[1]))

    
    

