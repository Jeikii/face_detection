import cv2 as cv

img = cv.imread('imgs/pp2.jpg')
cv.imshow('Group', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray group', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

print(f'Number of faces found = {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
  cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness = 3)

cv.imshow('Detected Faces', img)

cv.waitKey(0)