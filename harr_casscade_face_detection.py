import cv2 as cv 
import numpy as np 

img_file = "test_images\IMG-3002.JPG"
img = cv.imread(img_file)
width, height, channel = img.shape

if width < 1500 or height <1000:
    img = cv.resize(img, (1500, 1000), interpolation=cv.INTER_CUBIC)
else:
    img = cv.resize(img, (1500, 1000), interpolation=cv.INTER_AREA)

face_cascade = cv.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, 1.1, 5)

for x, y, w, h in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)

cv.imshow('casscade_results', img)
key = cv.waitKey(0)
if key == ord('q'):
    cv.destroyAllWindows()