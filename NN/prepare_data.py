import cv2
import numpy as np
import os
import glob

x = 128
y = 128
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

def preprocesing(image):
    # Operacja progowania globalnego
    ret, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow("progowanie", th1)
    #cv2.waitKey(0)

    return th1


# POBRANIE DANYCH Z PLIKÓW

train_name: str
for train_name in labels:
    cur_path = path + '/' + train_name
    cur_label = train_name

    for file in glob.glob(cur_path + "/*.jpg"):
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (x, y))
        gray = preprocesing(gray)
        # cv2.imshow("fota",gray)
        # cv2.waitKey()
        # print('juz')
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
        gray = cv2.resize(gray, (x, y))
        gray = preprocesing(gray)
        # cv2.imshow("fota",gray)
        # cv2.waitKey()
        # print('juz')
        test_data.append(gray)
        if cur_label == 'dobre':
            test_labels.append(1)
        else:
            test_labels.append(0)

valid_name: str
for valid_name in labels_valid:
    cur_path = path_valid + '/' + valid_name
    cur_label = valid_name

    for file in glob.glob(cur_path + "/*.jpg"):
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (x, y))
        gray = preprocesing(gray)
        # cv2.imshow("fota",gray)
        # cv2.waitKey()
        # print('juz')
        valid_data.append(gray)
        if cur_label == 'dobre':
            valid_labels.append(1)
        else:
            valid_labels.append(0)

# PRZYGOTOWANIE DANYCH
print(len(test_data))
train_labels = np.asarray(train_labels).astype('float32')
test_labels = np.asarray(test_labels).astype('float32')
valid_labels = np.asarray(valid_labels).astype('float32')

train_data = np.asarray(train_data)
train_data = train_data.reshape(152, x, y, 1)
train_data = train_data.astype('float32') / 255

test_data = np.asarray(test_data)
test_data = test_data.reshape(10, x, y, 1)
test_data = test_data.astype('float32') / 255

valid_data = np.asarray(valid_data)
valid_data = valid_data.reshape(27, x, y, 1)
valid_data = valid_data.astype('float32') / 255

# ZAPISANIE TABLIC

np.save('train_labels_1', train_labels)
np.save('test_labels_1', test_labels)
np.save('valid_labels_1', valid_labels)

np.save('train_data_1', train_data)
np.save('test_data_1', test_data)
np.save('valid_data_1', valid_data)

print(test_labels)
print(valid_labels)
print(test_data.shape)
