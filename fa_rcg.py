import cv2, sys, numpy, os

# The XML file contain a pre-trained face detection model. "A file that already knows the patterns commonly found in a human face."

haar_file = "/Users/arohanrudra/Desktop/OpenCV/L7-Facial Recognition/haarcascade_frontalface_default.xml"
datasets = "datasets"
sub_data = "Arohan"
base_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_path, 'datasets', sub_data)
if not os.path.isdir(path):
    os.makedirs(path)
(width,height) = (130, 100)
face_cascade = cv2.CascadeClassifier(haar_file)

webcam = cv2.VideoCapture(0)
count = 1
while count <= 30:
    (_, im) = webcam.read()
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = grey[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s/% s.png' % (path, count), face_resize)
    count += 1

    cv2.imshow("FR", im)
    key = cv2.waitKey(10)
    if key == 27:
        break
