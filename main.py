import datetime
import tensorflow as tf
from tensorflow import keras

import os
import math
import cv2
import urllib
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

url = 'https://wallpaperaccess.com/full/760289.jpg' # 514.02K kB (jpg)
response = requests.get(url)
img = Image.open(BytesIO(response.content))
data = np.array(img)
data = cv2.resize(data, (0, 0), fx = 0.1, fy = 0.1) # 126.80 kB (png)

cv2.imwrite("original.png", cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

#for i in range(2, 10):
for i in range(5, 10):
    L_FACT = 2**i +1
    print("L_FACT=", L_FACT)

    # Explode the input vector V into [sin(1.V),  sin(2.V), sin(3.V), sin(4.V), ... , sin((L-1).V)]
    def explode(xi, yi):
        v = np.array([xi, yi]).reshape((2,1)).T
        mul = np.arange(1, L_FACT).reshape((L_FACT-1, 1))
        #mul = np.power(2, np.arange(0, L_FACT)).reshape((L_FACT, 1))
        return  np.sin(np.matmul(mul, v)).flatten()

    # Dataset generation from the image
    X, y = [], []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            X.append( explode(i, j) )
            y.append( data[i,j] / 256.0 )

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=3214567)


    # Model Generation
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units = X.shape[1], activation = 'ReLU', input_shape = (X.shape[1], )))
    for i in range(0, int(math.ceil(math.log2(X.shape[1])))):
        units = max(3, X.shape[1] // 2**i)
        model.add(tf.keras.layers.Dense(units = units, activation = 'ReLU'))

    model.compile(optimizer = 'adam', loss=keras.metrics.mean_squared_error, metrics = ['mse'])
    model.summary()
    keras.utils.plot_model(
    model, to_file='imgs/model_' + str(L_FACT) + '.png', show_shapes=True, show_dtype=True,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
        layer_range=None, show_layer_activations=True
    )

    
    # Model Checkpoint Definition
    MODEL_SAVE_PATH = os.path.join('models', "model_" + str(L_FACT))
    CHECKPOINT_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'checkpoints')
    IMAGE_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'imgs')
    
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_SAVE_PATH, exist_ok=True)
    os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_SAVE_PATH,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            keys = list(logs.keys())
            if epoch%1==0:
                self.draw_img(self.model, data)

        def draw_img(self, model, data):
            xi = np.arange(0, data.shape[0])
            yi = np.arange(0, data.shape[1])
            
            res = np.zeros(data.shape, dtype=np.uint8)
        
            preds = np.transpose([np.tile(xi, len(yi)), np.repeat(yi, len(xi))])

            v = []
            for i in range(len(preds)):
                #tmpxi, tmpyi = np.round((preds[i][0]+1)*data.shape[0]/2.0), np.round((preds[i][1]+1)*data.shape[1]/2.0)
                tmpxi, tmpyi = np.round(preds[i][0]), np.round(preds[i][1])
                v.append(explode(tmpxi, tmpyi))
            v = np.array(v)

            mod_preds = model.predict(v)
            for i in range(len(preds)):
                px, py = np.round(preds[i][0]), np.round(preds[i][1])
                #px, py = int(np.round((preds[i][0]+1)*data.shape[0]/2.0)), int(np.round((preds[i][1]+1)*data.shape[1]/2.0))
                r, g, b = list(map(lambda tmp: int(round(tmp)), mod_preds[i] * 256.0))
                res[px,py] = r, g, b

            IMG_PATH = os.path.join(IMAGE_SAVE_PATH, str(datetime.datetime.now()) + '.png')
            #cv2.imwrite(IMG_PATH, res)
            cv2.imwrite(IMG_PATH, cv2.cvtColor(res, cv2.COLOR_BGR2RGB))


    # Model Training
    #model.fit(X, y, epochs=55, validation_data=(X_test, y_test), callbacks=[CustomCallback(), model_checkpoint_callback])
    #model.fit(X, y, epochs=55, validation_data=(X, y), callbacks=[CustomCallback(), model_checkpoint_callback])

    # Clean up
    del model, X, y, X_train, X_test, y_train, y_test
    print("-"*30)