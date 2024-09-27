import keras
import tensorflow as tf
import cv2
import sys
import pandas as pd
import numpy as np

# Embedding Size
embedding_size = 256
# Shape
shape = (175,175,1)
# Distance Function
def dist(a,b):
    return (np.linalg.norm(a-b))**2


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
    


model = keras.saving.load_model('./models/embeddings.keras')
database = pd.read_parquet('./database/database.parquet')
haarCascade = cv2.CascadeClassifier('./utils/face_detection.xml')



s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = 'Face Recognition'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)



while cv2.waitKey(100) != 27: # Escape
    has_frame, frame = source.read()
    if not has_frame:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection = haarCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4)

    for (x,y,w,h) in detection:
        face = image[y:y+h,x:x+w]
    inputs = cv2.resize(face, (shape[0],shape[1]), interpolation=cv2.INTER_CUBIC)
    inputs = np.array(inputs).reshape(-1,shape[0],shape[1],shape[2])/255.0
    outputs = model.predict([inputs])[0]

    minDist = 100
    identity = "Kaun Chutiya Hai"
    for index, row in database.iterrows():
        embeddings = row['Embeddings']
        person = row['Person']

        distance = dist(outputs, embeddings)
        if distance < minDist:
            identity = person
            minDist = distance

    threshold = 1.75
    if minDist > threshold:
        identity = "Kaun Chutiya Hai?"
    
    cv2.putText(frame, identity, (x,y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)