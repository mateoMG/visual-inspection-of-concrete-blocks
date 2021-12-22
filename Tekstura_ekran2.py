import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC

#odczyt danych zpliku odnośnie wymiarów 
plik2=open('pi/zmienne.txt')
i=0
dane=[0,0,0]
for linia in plik2:
    dane[i]=float(linia)
    i=i+1
pik=dane[0]
pik_y=dane[1]
pik_sre=dane[2]
plik2.close()
pow_pik=pik_sre*pik_sre

# Function which calculate dot product
def dot(tab_wag):
    h = 0
    wynik = 0
    while h<13:
        wynik = wynik + tab_wag[h]*clf.coef_[0][h]
        #print(wynik)
        h=h+1
    return wynik

# Function which calculate Haralicks texture features
def haralick(image):
     textures = mt.features.haralick(image)
     h_mean = textures.mean(axis=0)
     return h_mean

train_path = "train2/bloczki"
train_names = os.listdir(train_path)

train_features = []
train_labels = []

for train_name in train_names:
    cur_path = train_path + "/" + train_name
    cur_label = train_name
    i = 1
   
    for file in glob.glob(cur_path + "/*.jpg"):
        print ("Analizowanie zdjęcia - {} w {}".format(i,cur_label))
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = haralick(gray)
        train_features.append(features)
        train_labels.append(cur_label)        
        i = i+1
        
#print ("Training features: {}".format(np.array(train_features).shape))
#print ("Training labels: {}".format(np.array(train_labels).shape))
#Use LinearSVM as a machine learning model
clf = LinearSVC(random_state=0)
#Fit train data to model
clf.fit(train_features, train_labels)
file = "obraz_wyciety.png"

#Print classifier coef - optional
print(clf.coef_[0])
#print(clf.intercept_)

#Check your texture

print("Trwa ocena stanu tesktury")
image = cv2.imread(file)
image_predicted = image
(h, w) = image.shape[:2]
powierzchnia_calkowita=h*w

cellSizeYdir = (h // 10)

cellSizeXdir = (w // 10)
check = 0
image_zepsute=[]
image_wspolrzedne=[]
tablica_zepsute_piksele=[]
image_uszkodzone=[]
suma_kolor_dobry=0
for k in range (0,10,1):
    for j in range(0,10,1):
        cut = image[j*cellSizeYdir:(j+1)*cellSizeYdir, k*cellSizeXdir:(k+1)*cellSizeXdir]
        #cv2.imshow('cut',cut)
        gray = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        features = haralick(gray)
        x = round(dot(features),2)
        plik = open('dot_product.txt', 'a')
        plik.write("{},{}  coef -> {} file: {}\n".format(k,j,x,file))
        plik.close()
        prediction = clf.predict(features.reshape(1,-1))[0]
        if prediction == "zepsute":
            image_zepsute.insert(check,gray)
            cv2.imwrite('Zepsute/zepsute{}.png'.format(check), image_zepsute[check])
            image_wspolrzedne.insert(check,(k,j))
            check = check + 1
        else:
            zdjecie_dobre=bytearray(gray)
            suma_kolor_dobry=suma_kolor_dobry+sum(zdjecie_dobre)
            
        #cv2.waitKey(0)
kolor_sredni=suma_kolor_dobry/((100-check)*3888)
print(len(image_zepsute))
print(kolor_sredni)

image = cv2.resize(image,(960,540))
if check == 0:
    cv2.putText(image,"Bloczek dobry !!!", (15,45), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 3)
if check > 0:
    cv2.putText(image,"Bloczek zniszczony !!!", (15,45), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 3)
    prediction = "zepsute"
print("Decyzja {} -> {}".format(file,prediction))
image = cv2.resize(image,(250,250))

image_z_uszkodzeniami=cv2.imread(file,cv2.IMREAD_GRAYSCALE)

for i_1 in range(0,len(image_zepsute)-1):
    image_filter = cv2.bilateralFilter(image_zepsute[i_1],7, 80, 80)
    image_edges = cv2.Canny(image_filter,40,220)
    cv2.imwrite('krawedzie/zepsute{}.png'.format(i_1), image_edges)
    cv2.imwrite('filtr/zepsute{}.png'.format(i_1), image_filter)
    tablica_zdjecie=bytearray(image_filter)
    suma_pik=0
    for i_2 in range(0,len(tablica_zdjecie)-1):
        if tablica_zdjecie[i_2]<kolor_sredni-100:
            suma_pik=suma_pik+1
            tablica_zdjecie[i_2]=0
        else:
            tablica_zdjecie[i_2]=255
    tablica_zdjecie_nump=np.array(tablica_zdjecie)
    image_uszkodzone.insert(i_1,tablica_zdjecie_nump.reshape(cellSizeYdir,cellSizeXdir))
    cv2.imwrite('uszkodzone/uszko{}.png'.format(i_1), image_uszkodzone[i_1])
    (posy,posx)=image_wspolrzedne[i_1]
    image_z_uszkodzeniami[posx*cellSizeYdir:(posx+1)*cellSizeYdir, posy*cellSizeXdir:(posy+1)*cellSizeXdir]=image_uszkodzone[i_1]

    tablica_zepsute_piksele.insert(i_1,suma_pik)
cv2.imshow("canny",image_z_uszkodzeniami)
cv2.waitKey(0)
zepsute_piksele=sum(tablica_zepsute_piksele)
zepsute_piksele_mm=zepsute_piksele*pow_pik
wzgledna_powierzchnia_uszkodzen=zepsute_piksele/powierzchnia_calkowita*100
print(zepsute_piksele)
print(zepsute_piksele_mm)
print(wzgledna_powierzchnia_uszkodzen)
#cv2.imshow("Ocena",image)

cv2.waitKey(16000)
if prediction == "dobre":
    print("bloczek dobry")
if prediction == "zepsute":
    path = 'pdamage'
    cv2.imwrite(os.path.join(path , 'texture.jpg'), image_predicted)
#Finish
cv2.destroyAllWindows()
print("koniec") 
