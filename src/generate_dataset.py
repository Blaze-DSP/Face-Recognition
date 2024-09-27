import os
import cv2
import numpy as np

# Input Shape
shape = (175,175,1)

# To Detect Faces
haarCascade = cv2.CascadeClassifier('./utils/face_detection.xml')
def faceDetect(img):
    facesDetected = haarCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)

    face = []
    for (x,y,w,h) in facesDetected:
        face = img[y:y+h, x:x+w]
    
    return np.array(face)

images = []
persons = []

dir = './data'
for person in os.listdir(dir):
    for image in os.listdir(os.path.join(dir, person)):
        img = os.path.join(dir,person,image)
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)

        img = faceDetect(img)

        if(img.shape[0]==0):
            continue
        
        img = cv2.resize(img, (shape[0],shape[1]), interpolation=cv2.INTER_CUBIC)
        images.append(img)
        persons.append(person)
        
    print(person)

images = np.array(images).reshape(-1, shape[0], shape[1], shape[2])/255.0
persons = np.array(persons).reshape(-1,1)

print(images.shape)
print(persons.shape)

dir = './dataset'
dataset = 3
np.save(f'{dir}/d{dataset}_images.npy', images)
np.save(f'{dir}/d{dataset}_persons.npy',persons)