from picamera import PiCamera

k=100
kamera = PiCamera()
kamera.resolution = (12*k, 9*k)
kamera.rotation = 360
kamera.capture('/home/pi/Desktop/pi/bloczek.jpg')
#kamera.close()




    
