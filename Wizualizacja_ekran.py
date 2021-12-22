import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC


# Function which calculate Haralicks texture features
def haralick(image):
     textures = mt.features.haralick(image)
     h_mean = textures.mean(axis=0)
     return h_mean

train_path = "pi/train2/bloczki"
train_names = os.listdir(train_path)

train_features = []
train_labels = []

for train_name in train_names:
    cur_path = train_path + "/" + train_name
    cur_label = train_name
    i = 1
   
    for file in glob.glob(cur_path + "/*.jpg"):
        #print ("Analizowanie zdjęcia - {} w {}".format(i,cur_label))
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = haralick(gray)
        train_features.append(features)
        train_labels.append(cur_label)        
        i = i+1
        
clf = LinearSVC(random_state=0)
#Fit train data to model
clf.fit(train_features, train_labels)
test_path = "pi/obraz_wyciety.png"

print("Trwa przygotowanie wizualizacji uszkodzen")
image_predicted = cv2.imread(test_path)
image_predicted = cv2.resize(image_predicted,(960,540))
(h, w) = image_predicted.shape[:2]
cellSizeYdir = (h // 20)
cellSizeXdir = (w // 20)
for x in range(0,w,cellSizeXdir):
    cv2.line(image_predicted, (x, 0), (x, h), (255, 0, 0), 1)
for y in range(0, h, cellSizeYdir):
    cv2.line(image_predicted, (0, y), (w, y), (255, 0, 0), 1)
cv2.line(image_predicted, (0, h - 1 ), (w, h - 1 ), (255, 0, 0), 1)
cv2.line(image_predicted, (w - 1, 0), (w - 1, h - 1), (255, 0, 0), 1)
#cv2.imshow("1",image_predicted)
number = 0
for k in range (0,20,1):
    for j in range(0,20,1):
        cut = image_predicted[j*cellSizeYdir:(j+1)*cellSizeYdir, k*cellSizeXdir:(k+1)*cellSizeXdir]
        gray_2 = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        features_2 = haralick(gray_2)
        prediction_2 = clf.predict(features_2.reshape(1,-1))[0]
        if prediction_2 == "zepsute":
            number = number + 1
            cv2.putText(image_predicted,"-", (((k+1)*cellSizeXdir-24),((j+1)*cellSizeYdir-14)), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0,0,255), 3) 
        if prediction_2 == "dobre":
            cv2.putText(image_predicted,"-", (((k+1)*cellSizeXdir-24),((j+1)*cellSizeYdir-14)), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0,255,0), 3)
procent = (number / 4)
plik = open('pi/procent.txt', 'a')
plik.write("{} -> {}% uszkodzen \n".format(file,procent))
plik.close()
#image_predicted = cv2.resize(image_predicted,(250,250))
cv2.imshow("Zaznaczenie uszkodzen",image_predicted)
cv2.waitKey(16000)
#Finish
cv2.destroyAllWindows()
print("koniec") 