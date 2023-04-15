#testing accuracy on test dataset
import tensorflow as tf

import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = tf.keras.models.load_model('my_model')
# model.load('my_model')

y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values
data=[]
for img in imgs:
   image = Image.open(img)
   image = image.resize((30,30))
   data.append(np.array(image))
X_test=np.array(data)
pred = np.argmax(model.predict(X_test),axis=1)

print(accuracy_score(labels, pred))