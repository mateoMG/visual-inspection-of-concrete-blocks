from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

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


test_data = np.load('test_data_1.npy')
test_labels = np.load('test_labels_1.npy')

train_data = np.load('train_data_1.npy')
train_labels = np.load('train_labels_1.npy')

valid_data = np.load('valid_data_1.npy')
valid_labels = np.load('valid_labels_1.npy')

print (' SIEC NEURONOWA: ')

neuron = models.Sequential()
neuron.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(128,128,1)))
neuron.add(layers.MaxPooling2D((2,2)))
neuron.add(layers.Conv2D(32, (3,3), activation='relu'))
neuron.add(layers.MaxPooling2D((2,2)))
neuron.add(layers.Conv2D(32, (3,3), activation='relu'))
neuron.add(layers.Flatten())
neuron.add(layers.Dense(1024, activation='relu'))
neuron.add(layers.Dense(512, activation='relu'))
neuron.add(layers.Dense(32, activation='relu'))
neuron.add(layers.Dense(1, activation='sigmoid'))

neuron.compile(optimizer='rmsprop',loss = 'binary_crossentropy' ,metrics=['accuracy'])
#print(neuron.summary())
history = neuron.fit(train_data, train_labels, epochs = 15, batch_size=2, validation_data=(valid_data,valid_labels))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_variable = range(1, len(acc)+1)

wyniki = neuron.evaluate(test_data,test_labels)
print(wyniki)

#zapisanie modelu
neuron.save("model.h5")
# RYSOWANIE WYKRESOW

predictions = neuron.predict(test_data)
print(predictions)

strata_trenowania_walidacji(loss,val_loss,epochs_variable)

dokladnosc_trenowania_walidacji(acc,val_acc,epochs_variable)

print('koniec')