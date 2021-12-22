import cv2
import numpy as np
import os
import glob
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# FUNKCJE WYKRESY

def strata_trenowania_walidacji (x,y,e):

    plt.plot(e, x, 'bo', label='Strata trenowania')
    plt.plot(e, y, 'b', label='Strata walidacji')
    plt.title('Strata trenowania i walidacji')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.legend()

    plt.show()


def dokladnosc_trenowania_walidacji(x, y, e):
    plt.plot(e, x, 'bo', label='Dokladnosc trenowania')
    plt.plot(e, y, 'b', label='Dokladnosc walidacji')
    plt.title('Dokladnosc trenowania i walidacji')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.legend()

    plt.show()

# ZMIENNE DLA DANYCH TRENINGOWYCH I TESTOWYCH
x = 64
y = 64
path = 'dataset'
labels = os.listdir(path)

path_test = 'testdata'
labels_test = os.listdir(path_test)

path_valid = 'valid'
labels_valid = os.listdir(path_valid)

train_data = []
train_labels = []

test_data = []
test_labels = []

valid_data = []
valid_labels = []

#POBRANIE DANYCH Z PLIKÓW

train_name: str
for train_name in labels:
    cur_path = path + '/' + train_name
    cur_label = train_name

    for file in glob.glob(cur_path + "/*.jpg"):
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(x,y))
        #cv2.imshow("fota",gray)
        #cv2.waitKey()
        #print('juz')
        train_data.append(gray)
        if cur_label == 'dobre':
            train_labels.append(1)
        else:
            train_labels.append(0)

test_name: str
for test_name in labels_test:
    cur_path = path_test + '/' + test_name
    cur_label = test_name

    for file in glob.glob(cur_path + "/*.jpg"):
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(x,y))
        #cv2.imshow("fota",gray)
        #cv2.waitKey()
        #print('juz')
        test_data.append(gray)
        if cur_label == 'dobre':
            test_labels.append(1)
        else:
            test_labels.append(0)

for valid_name in labels_valid:
    cur_path = path_valid + '/' + valid_name
    cur_label = valid_name

    for file in glob.glob(cur_path + "/*.jpg"):
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(x,y))
        #cv2.imshow("fota",gray)
        #cv2.waitKey()
        #print('juz')
        valid_data.append(gray)
        if cur_label == 'dobre':
            valid_labels.append(1)
        else:
            valid_labels.append(0)

# PRZYGOTOWANIE DANYCH

train_labels = np.asarray(train_labels).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')
valid_labels = np.asarray(valid_labels).astype('float32')

train_data = np.asarray(train_data)
train_data = train_data.reshape(86,x,y,1)
train_data = train_data.astype('float32')/255

test_data = np.asarray(test_data)
test_data = test_data.reshape(10,x,y,1)
test_data = test_data.astype('float32')/255

valid_data = np.asarray(valid_data)
valid_data = valid_data.reshape(29,x,y,1)
valid_data = valid_data.astype('float32')/255

print(train_data.shape)
print(test_data.shape)
print(train_labels.shape)
print(test_labels.shape)
print(valid_data.shape)
print(valid_labels.shape)

# BUDUJEMY SIEC NEURONOWA

print (' SIEC NEURONOWA: ')

neuron = models.Sequential()
neuron.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(x,y,1)))
neuron.add(layers.MaxPooling2D((2,2)))
neuron.add(layers.Conv2D(64, (3,3), activation='relu'))
neuron.add(layers.MaxPooling2D((2,2)))
neuron.add(layers.Conv2D(64, (3,3), activation='relu'))
neuron.add(layers.Flatten())
neuron.add(layers.Dense(512, activation='relu'))
neuron.add(layers.Dense(64, activation='relu'))
neuron.add(layers.Dense(1, activation='sigmoid'))

neuron.compile(optimizer='rmsprop',loss = 'binary_crossentropy' ,metrics=['accuracy'])
#print(neuron.summary())
history = neuron.fit(train_data, train_labels, epochs = 20, batch_size=2, validation_data=(valid_data,valid_labels))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_variable = range(1, len(acc)+1)

wyniki = neuron.evaluate(test_data,test_labels)
print(wyniki)

# RYSOWANIE WYKRESOW

strata_trenowania_walidacji(loss,val_loss,epochs_variable)

dokladnosc_trenowania_walidacji(acc,val_acc,epochs_variable)

print('koniec')