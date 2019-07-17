import cv2 as cv
from mtcnn.mtcnn import MTCNN

img_path = "test_images/mink_minkesh/IMG_2628.JPG"
img = cv.imread(img_path)
#img = cv.resize(img, (800, 800), interpolation=cv.INTER_AREA)
detector = MTCNN()
results = detector.detect_faces(img)

'''
# For visualization
for r in results:
    face_box = r['box']
    x, y, width, height = face_box
    cv.rectangle(img, (x, y), (x+width, y+height), color=(0, 0, 255), thickness=2)
'''




