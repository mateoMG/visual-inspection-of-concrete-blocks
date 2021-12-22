from keras.models import load_model
import numpy as np
import os
import glob
import cv2

print("TEST")

test_data = np.load('test_data_1.npy')
test_labels = np.load('test_labels_1.npy')

model = load_model("model_01.h5")
model.summary()
model.evaluate(test_data,test_labels)

wyniki = model.predict(test_data)

print(wyniki)
#print(wyniki.shape)

train_path = "testdata"
train_names = os.listdir(train_path)

print(train_names)

i = 1
for train_name in train_names:
    cur_path = train_path + "/" + train_name
    cur_label = train_name


    for file in glob.glob(cur_path + "/*.jpg"):
        image = cv2.imread(file)
        image = cv2.resize(image, (960, 540))
        print(wyniki[i - 1][0])
        if (wyniki[i - 1][0])>0.5:
            cv2.putText(image, "Nawierzchnia dobra !", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            cv2.putText(image, "Uwaga zniszczona nawierzchnia !!!", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imshow("Test", image)
        cv2.waitKey()
        i = i + 1


