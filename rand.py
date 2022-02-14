from operator import mod
import os
import math
from statistics import mode
import cv2
import urllib
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import datetime
import tensorflow as tf
from tensorflow import keras

import sys

sys.path.append(os.path.join(os.path.curdir, 'kitti-utils'))
from data_parser import KittiDataParser
from calibration import Calibration

KITTI_PATH = "/mnt/HDD/home/aditya/kitti/kitti_data/"
KITTI_EXAMPLE = os.path.join(KITTI_PATH, "2011_09_26/2011_09_26_drive_0001_sync/")
KITTI_CALIB_CAM2CAM = os.path.join(KITTI_PATH, "2011_09_26/calib_cam_to_cam.txt")
KITTI_CALIB_IMU2VEL = os.path.join(KITTI_PATH, "2011_09_26/calib_imu_to_velo.txt")
KITTI_CALIB_VEL2CAM = os.path.join(KITTI_PATH, "2011_09_26/calib_velo_to_cam.txt")

kitti = KittiDataParser(KITTI_EXAMPLE, file_mode=1)
calib = Calibration(KITTI_CALIB_CAM2CAM)

img_arr = kitti.load_image("0000000000")

for i in range(len(img_arr)):
    img_arr[i] = cv2.resize(img_arr[i].copy(), (0, 0), fx = 0.1, fy = 0.1) 
    #img_arr[i] = cv2.cvtColor(img_arr[i].copy(), cv2.COLOR_BGR2RGB)

#cv2.imwrite("original.png", cv2.cvtColor(data, cv2.COLOR_BGR2RGB))

def run_training(SIGMA, NUM):
    #SIGMA = i
    #NUM = j
    print("SIGMA=", SIGMA, "NUM=",NUM)
    np.random.seed(0)
    B = np.random.normal(0, SIGMA**0.5, size=NUM).reshape((NUM, 1))
    #print(B)

    # Explode the input vector v into [sin(B.v),  cos(B.v)] such that B is normal distribution N(0, SIGMA) with NUM elements
    def explode(P, B=B):
        xi, yi, zi = P[0,0], P[1,0], P[2,0]
        v = np.array([xi, yi, zi]).reshape((3,1))
        
        mul = np.matmul(B, v.T)
        
        s = np.sin(mul).flatten()
        c = np.cos(mul).flatten()
        return np.concatenate((s,c))

    # Dataset generation from the image
    X, y = [], []
    for img_i in range(len(img_arr)):
        cam_name = '0' + str(img_i)
        R = calib.calib['R_' + cam_name].reshape((3,3))
        T = calib.calib['T_' + cam_name].reshape((3,1))
        data = img_arr[img_i]
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                point = np.array([i, j, 0]).reshape((3,1))
                X.append( explode(np.matmul(R, point) + T) )
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
    """
    keras.utils.plot_model( model, to_file='imgs/model_'+ str(SIGMA)) + "_" + str(NUM) + '.png', show_shapes=True, show_dtype=True,
        show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96,
        layer_range=None, show_layer_activations=True
    )
    """
    
    # Model Checkpoint Definition
    MODEL_NAME = "model_" + str(SIGMA)+ "_" + str(NUM)
    MODEL_SAVE_PATH = os.path.join('models',  MODEL_NAME)
    CHECKPOINT_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'checkpoints')
    MODEL_H5_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, MODEL_NAME + ".h5")
    IMAGE_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'imgs')
    B_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, 'B.npy')
    
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(CHECKPOINT_SAVE_PATH, exist_ok=True)
    os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)

    np.save(B_SAVE_PATH, B)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_SAVE_PATH,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    class CustomCallback(keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.best_val_loss = float('inf')
        def on_epoch_end(self, epoch, logs=None):

            if logs['val_loss'] < self.best_val_loss:
                self.best_val_loss = logs['val_loss']
                self.model.save(MODEL_H5_SAVE_PATH, overwrite=True)

            if epoch%1==0:
                for img_i in range(len(img_arr)):
                    cam_name = '0' + str(img_i)
                    R = calib.calib['R_' + cam_name].reshape((3,3))
                    T = calib.calib['T_' + cam_name].reshape((3,1))
                    data = img_arr[img_i]
                    
                    self.draw_img(self.model, data, R, T, cam_name)

        def draw_img(self, model, data):
            xi = np.arange(0, data.shape[0])
            yi = np.arange(0, data.shape[1])
            
            res = np.zeros(data.shape, dtype=np.uint8)
        
            preds = np.transpose([np.tile(xi, len(yi)), np.repeat(yi, len(xi))])


            v = []
            for i in range(len(preds)):
                #tmpxi, tmpyi = np.round((preds[i][0]+1)*data.shape[0]/2.0), np.round((preds[i][1]+1)*data.shape[1]/2.0)
                tmpxi, tmpyi = np.round(preds[i][0]), np.round(preds[i][1])
                point = np.array([tmpxi, tmpyi, 0]).reshape((3,1))
                v.append( explode(np.matmul(R, point) + T) )
                #v.append(explode(tmpxi, tmpyi))
            v = np.array(v)

            mod_preds = model.predict(v)
            for i in range(len(preds)):
                px, py = np.round(preds[i][0]), np.round(preds[i][1])
                #px, py = int(np.round((preds[i][0]+1)*data.shape[0]/2.0)), int(np.round((preds[i][1]+1)*data.shape[1]/2.0))
                r, g, b = list(map(lambda tmp: int(round(tmp)), mod_preds[i] * 256.0))
                res[px,py] = r, g, b

            IMG_PATH = os.path.join(IMAGE_SAVE_PATH, str(datetime.datetime.now()) + '.png')
            #cv2.imwrite(IMG_PATH, res)
            #cv2.imwrite(IMG_PATH, cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            cv2.imwrite(IMG_PATH, cv2.cvtColor(res.copy(), cv2.COLOR_BGR2RGB))


    # Model Training
    #model.fit(X, y, epochs=55, validation_data=(X_test, y_test), callbacks=[CustomCallback(), model_checkpoint_callback])
    model.fit(X, y, epochs=25, validation_data=(X, y), callbacks=[CustomCallback(), model_checkpoint_callback])

    # Clean up
    # del model, X, y, X_train, X_test, y_train, y_test
    print("-"*30)
    """
    Usage:
    p = [(x1,y1), (x2, y2), ... , (xn, yn)] # list of query points
    v = list(map(lambda pi: explode(pi[0], pi[1], B), p)) # input vector
    predictions = model.predict(v) # Model predictions
    """
    return model, B, explode

if __name__=="__main__":
    run_training(200, 256)
    exit()
    for SIGMA in range(1, 500, 10):
        for NUM in range(5, 40, 5):
            pass