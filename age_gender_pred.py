import cv2 as cv 
import numpy as np 
import face_detection as f_d

img_path = "test_images\group_1.jpg"
img = cv.imread(img_path)
width, height, channel = img.shape
#if not (width < 100 or height < 800):
if width < 1500 or height <1000:
    img = cv.resize(img, (1500, 1000), interpolation=cv.INTER_CUBIC)
else:
    img = cv.resize(img, (1500, 1000), interpolation=cv.INTER_AREA)
f_d_results = f_d.detector.detect_faces(img)

gender_proto = 'weight_files/gender_deploy.prototxt.txt'
gender_model = 'weight_files/gender_net.caffemodel'

age_proto = 'weight_files/age_deploy.prototxt.txt'
age_model = 'weight_files/age_net.caffemodel'

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(age_model, age_proto)
genderNet = cv.dnn.readNet(gender_model, gender_proto)

if not gender_proto:
    print('There is no face detected')

padding = 20
for i, r in enumerate(f_d_results):
    face_box = r['box']
    x, y, width, height = face_box
    #cropped_face = img[y:y+height, x:x+width].copy() 
    face = img[max(0, y-padding):min(y+height+padding, img.shape[0]-1), max(0, x-padding):min(x+width+padding, img.shape[1]-1)]
    #cv.imshow(f'face: {i+1}', cropped_face)

    blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    
    cv.rectangle(img, (x, y), (x+width, y+height), color=(255, 0, 0), thickness= 2)

    if gender == 'Female':
        gender = 'F'
    else:
        gender = 'M'
    cv.putText(img, f'{gender}, {age}', (x, y), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)


cv.imshow('face_detection', img)
key = cv.waitKey(0)
if key == ord('s'):
    cv.imwrite('test_results/out_1.jpg', img)
    cv.destroyAllWindows() 
if key == ord('q'):
    cv.destroyAllWindows()

