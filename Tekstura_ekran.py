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

def wyznacz_sasiad(image,pozycja, max_x, max_y, tab):
    index_uszko_1=0
    index_uszko_2=0
    stan_l_s=0
    stan_p_s=0
    stan_g_s=0
    stan_d_s=0
    for  i_g in range(0,max_x):
        if image[0,i_g]==0:
            index_uszko_1=index_uszko_1+1
            if index_uszko_1==5:
                stan_g_s=1
        else:
            index_uszko_1=0
        if image[max_y,i_g]==0:
            index_uszko_2=index_uszko_2+1
            if index_uszko_2==5:
                stan_d_s=1
        else:
            index_uszko_2=0
    index_uszko_1=0
    index_uszko_2=0  
    for  i_s in range(0,max_y):
        if image[i_s,0]==0:
            index_uszko_1=index_uszko_1+1
            if index_uszko_1==5:
                stan_l_s=1
        else:
            index_uszko_1=0
        if image[i_s,max_x]==0:
            index_uszko_2=index_uszko_2+1
            if index_uszko_2==5:
                stan_p_s=1
        else:
            index_uszko_2=0
    tab.insert(pozycja,[stan_l_s,stan_g_s,stan_p_s,stan_d_s])
def wytnij_zdjecie_funkcja(img, poz_y,poz_x,krok_y,krok_x,img_zep,img_wspol,img_uszko,img_z_uszko,poz_zdjecie, max_suma,tab_suma):
    cut = img[poz_y*krok_y:(poz_y+1)*krok_y, poz_x*krok_x:(poz_x+1)*krok_x]
    
   
    img_zep.insert(poz_zdjecie,cut)
    cv2.imwrite('pi/Zepsute/zepsute{}.png'.format(poz_zdjecie), img_zep[poz_zdjecie])
    img_wspol.insert(poz_zdjecie,(poz_y,poz_x))
    image_filter = cv2.bilateralFilter(img_zep[poz_zdjecie],3, 80, 80)
    
    tablica_zdjecie=bytearray(image_filter)
    suma_pik=0
    for i_2 in range(0,len(tablica_zdjecie)-1):
        if tablica_zdjecie[i_2]<max_suma-30:
            suma_pik=suma_pik+1
            tablica_zdjecie[i_2]=0
        else:
            tablica_zdjecie[i_2]=255
    tablica_zdjecie_nump=np.array(tablica_zdjecie)
    img_uszko.insert(poz_zdjecie,tablica_zdjecie_nump.reshape(krok_y,krok_x))

    cv2.imwrite('pi/uszkodzone/uszko{}.png'.format(poz_zdjecie), img_uszko[poz_zdjecie])
    img_z_uszko[poz_y*krok_y:(poz_y+1)*krok_y, poz_x*krok_x:(poz_x+1)*krok_x]=img_uszko[poz_zdjecie]
    tab_suma.insert(poz_zdjecie,suma_pik)    
    
    

train_path = "pi/train2/bloczki"
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
file = "pi/obraz_wyciety.png"

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
print(cellSizeYdir)

cellSizeXdir = (w // 10)
check = 0
image_zepsute=[]
image_wspolrzedne=[]
tablica_zepsute_piksele=[]
tab_uszko_sasiad=[]

image_uszkodzone=[]
suma_kolor_dobry=0
plik = open('pi/dot_product.txt', 'w')
plik.write("")
plik.close()
for k in range (0,10,1):
    for j in range(0,10,1):
        cut = image[j*cellSizeYdir:(j+1)*cellSizeYdir, k*cellSizeXdir:(k+1)*cellSizeXdir]
        #cv2.imshow('cut',cut)
        gray = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('pi/podzial/dobre_j{}_k{}.png'.format(j,k), gray)
        features = haralick(gray)
        x = round(dot(features),2)
        plik = open('pi/dot_product.txt', 'a')
        plik.write("{},{}  coef -> {} file: {}\n".format(k,j,x,file))
        plik.close()
        prediction = clf.predict(features.reshape(1,-1))[0]
        if prediction == "zepsute":
            image_zepsute.insert(check,gray)
            cv2.imwrite('pi/Zepsute/zepsute{}.png'.format(check), image_zepsute[check])
            image_wspolrzedne.insert(check,(j,k))
            check = check + 1
        else:
            zdjecie_dobre=bytearray(gray)
            suma_kolor_dobry=suma_kolor_dobry+sum(zdjecie_dobre)
            
        #cv2.waitKey(0)
kolor_sredni=suma_kolor_dobry/((100-check)*cellSizeXdir*cellSizeYdir)

            
            

image = cv2.resize(image,(960,540))
if check == 0:
    cv2.putText(image,"Bloczek dobry !!!", (15,45), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 3)
if check > 0:
    cv2.putText(image,"Bloczek zniszczony !!!", (15,45), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255), 3)
    prediction = "zepsute"
print("Decyzja {} -> {}".format(file,prediction))
image = cv2.resize(image,(250,250))



image_z_uszkodzeniami=cv2.imread(file,cv2.IMREAD_GRAYSCALE)

for i_1 in range(0,len(image_zepsute)):
    image_filter = cv2.bilateralFilter(image_zepsute[i_1],3, 80, 80)
    image_edges = cv2.Canny(image_filter,40,220)
    cv2.imwrite('krawedzie/zepsute{}.png'.format(i_1), image_edges)
    cv2.imwrite('filtr/zepsute{}.png'.format(i_1), image_filter)
    tablica_zdjecie=bytearray(image_filter)
    suma_pik=0
    for i_2 in range(0,len(tablica_zdjecie)-1):
        if tablica_zdjecie[i_2]<kolor_sredni-30:
            suma_pik=suma_pik+1
            tablica_zdjecie[i_2]=0
        else:
            tablica_zdjecie[i_2]=255
    tablica_zdjecie_nump=np.array(tablica_zdjecie)
    image_uszkodzone.insert(i_1,tablica_zdjecie_nump.reshape(cellSizeYdir,cellSizeXdir))
    cv2.imwrite('pi/uszkodzone/uszko{}.png'.format(i_1), image_uszkodzone[i_1])
    (posy,posx)=image_wspolrzedne[i_1]
    image_z_uszkodzeniami[posy*cellSizeYdir:(posy+1)*cellSizeYdir, posx*cellSizeXdir:(posx+1)*cellSizeXdir]=image_uszkodzone[i_1]

    tablica_zepsute_piksele.insert(i_1,suma_pik)
i_uszko=0
image_1=cv2.imread(file,cv2.IMREAD_GRAYSCALE)

while (i_uszko<len(image_uszkodzone)):
    wyznacz_sasiad(image_uszkodzone[i_uszko],i_uszko,cellSizeXdir-1,cellSizeYdir-1,tab_uszko_sasiad)
    
    
    i_uszko_2=0
    (pos_y_spr, pos_x_spr)=image_wspolrzedne[i_uszko]
    
    if tab_uszko_sasiad[i_uszko][0]==1:
        
        
        wytnij_zdjecie=True
        while (i_uszko_2<len(image_uszkodzone)):
            (posy,posx)=image_wspolrzedne[i_uszko_2]
            if (pos_x_spr==0 or pos_y_spr==posy and pos_x_spr==posx+1):
                wytnij_zdjecie=False
            i_uszko_2=i_uszko_2+1     
        if wytnij_zdjecie:
            
            pozycja=len(image_uszkodzone)
            wytnij_zdjecie_funkcja(image_1, pos_y_spr,pos_x_spr-1,cellSizeYdir,cellSizeXdir,image_zepsute,image_wspolrzedne,image_uszkodzone,image_z_uszkodzeniami,pozycja,kolor_sredni,tablica_zepsute_piksele)
       
    if tab_uszko_sasiad[i_uszko][1]==1:
        i_uszko_2=0 
        
        wytnij_zdjecie=True
        while (i_uszko_2<len(image_uszkodzone)):
            (posy,posx)=image_wspolrzedne[i_uszko_2]
            if (pos_y_spr==0 or pos_y_spr==posy+1 and pos_x_spr==posx):
                wytnij_zdjecie=False
            i_uszko_2=i_uszko_2+1     
        if wytnij_zdjecie:
            
            pozycja=len(image_uszkodzone)
            wytnij_zdjecie_funkcja(image_1, pos_y_spr-1,pos_x_spr,cellSizeYdir,cellSizeXdir,image_zepsute,image_wspolrzedne,image_uszkodzone,image_z_uszkodzeniami,pozycja,kolor_sredni,tablica_zepsute_piksele)
                     
    if tab_uszko_sasiad[i_uszko][2]==1:
        i_uszko_2=0 
        
        wytnij_zdjecie=True
        while (i_uszko_2<len(image_uszkodzone)):
            (posy,posx)=image_wspolrzedne[i_uszko_2]
            if (pos_x_spr==9 or pos_y_spr==posy and pos_x_spr==posx-1):
                wytnij_zdjecie=False
            i_uszko_2=i_uszko_2+1     
        if wytnij_zdjecie:
            
            pozycja=len(image_uszkodzone)
            wytnij_zdjecie_funkcja(image_1, pos_y_spr,pos_x_spr+1,cellSizeYdir,cellSizeXdir,image_zepsute,image_wspolrzedne,image_uszkodzone,image_z_uszkodzeniami,pozycja,kolor_sredni,tablica_zepsute_piksele)
                     
    if tab_uszko_sasiad[i_uszko][3]==1:
        i_uszko_2=0 
        
        wytnij_zdjecie=True
        while (i_uszko_2<len(image_uszkodzone)):
            (posy,posx)=image_wspolrzedne[i_uszko_2]
            if (pos_y_spr==9 or pos_y_spr==posy+1 and pos_x_spr==posx):
                wytnij_zdjecie=False
            i_uszko_2=i_uszko_2+1     
        if wytnij_zdjecie:
            
            pozycja=len(image_uszkodzone)
            wytnij_zdjecie_funkcja(image_1, pos_y_spr+1,pos_x_spr,cellSizeYdir,cellSizeXdir,image_zepsute,image_wspolrzedne,image_uszkodzone,image_z_uszkodzeniami,pozycja,kolor_sredni,tablica_zepsute_piksele)
                     
                                           
        
            
           
    i_uszko=i_uszko+1

cv2.imwrite('pi/Obraz_uszkodzen_duzy.png', image_z_uszkodzeniami)

zepsute_piksele=sum(tablica_zepsute_piksele)
zepsute_piksele_mm=round(zepsute_piksele*pow_pik,2)
wzgledna_powierzchnia_uszkodzen=round(zepsute_piksele/powierzchnia_calkowita*100,2)

print(zepsute_piksele_mm)
print(wzgledna_powierzchnia_uszkodzen)
image_wymiary=cv2.imread(file)
stala_zdjecie=h/320
image_wymiary = cv2.resize(image_wymiary,(int(w/stala_zdjecie),320))
cv2.putText(image_wymiary,"Powierzchnia", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
cv2.putText(image_wymiary,"{} mm".format(zepsute_piksele_mm), (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
cv2.putText(image_wymiary,"{} %".format(wzgledna_powierzchnia_uszkodzen), (5,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
cv2.imwrite('pi/Obraz_z_powierzchni.png', image_wymiary)
plik = open('pi/powierzchnia.txt', 'w')
plik.write("Powierzchnia uszkodzeń w mm^2: {}\nPowierzchnia uszkodzeń w procentach {}\n".format(zepsute_piksele_mm,wzgledna_powierzchnia_uszkodzen))
   
    
plik.close()
cv2.imshow("Ocena",image)

cv2.waitKey(16000)
if prediction == "dobre":
    print("bloczek dobry")
if prediction == "zepsute":
    path = 'pdamage'
    cv2.imwrite(os.path.join(path , 'texture.jpg'), image_predicted)
#Finish
cv2.destroyAllWindows()
print("koniec") 
