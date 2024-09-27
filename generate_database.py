import warnings
warnings.filterwarnings('ignore')



import cv2
import numpy as np

haar_cascade = cv2.CascadeClassifier('./utils/face_detection.xml')
# To Detect Faces
def face_detect(img):
    faces_detected = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)

    face = []
    for (x,y,w,h) in faces_detected:
        face = img[y:y+h, x:x+w]
    
    return np.array(face)



import keras
import tensorflow as tf

# Embedding Size
embedding_size = 256

@keras.saving.register_keras_serializable()
class ConvBlock(keras.layers.Layer):
    def __init__(self, filters,regularizer=None,**kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.regularizer = regularizer
        self.conv1 = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv2 = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv3 = keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv4 = keras.layers.Conv2D(filters=int(filters/2), kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizer)
        self.pool = keras.layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x = keras.layers.Add()([x1, x2])
        x = self.conv3(x)
        x = self.conv4(x)
        return self.pool(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'regularizer': self.regularizer})
        
        return config

@keras.saving.register_keras_serializable()
# Defining Triplet Loss Fucntion
class TripletLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, name='triplet_loss',**kwargs):
        super(TripletLoss, self).__init__(name=name,**kwargs)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        anchor_embed = y_pred[:, :embedding_size]
        positive_embed = y_pred[:, embedding_size:embedding_size*2]
        negative_embed = y_pred[:, embedding_size*2:]

        positive_dist = tf.reduce_sum(tf.square(anchor_embed - positive_embed), axis=-1)
        negative_dist = tf.reduce_sum(tf.square(anchor_embed - negative_embed), axis=-1)
        
        loss = tf.maximum(positive_dist - negative_dist + self.alpha, 0.0)
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha})
        
        return config
    
keras.saving.get_custom_objects()

model = keras.saving.load_model('./models/embeddings.keras')



import pandas as pd
database = pd.DataFrame(columns=['Person','Embeddings'])



import os
# Input Shape
shape = (175,175,1)
path = './database/images'
for person in os.listdir(path):
    for img in os.listdir(os.path.join(path,person)):
        img_path = os.path.join(path,person,img)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        image = face_detect(image)

        if(image.shape[0]==0):
            print(person, "Error")
            continue

        image = cv2.resize(image, (shape[0],shape[1]), interpolation=cv2.INTER_CUBIC)
        image = np.array(image).reshape(-1,shape[0],shape[1],shape[2])/255.0

        embeddings = model.predict(image)

        database.loc[len(database)] = [person, embeddings[0]]

        print(person, "Done")




database.to_parquet('./database/database.parquet')
database.head()