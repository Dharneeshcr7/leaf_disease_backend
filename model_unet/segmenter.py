import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from matplotlib import pyplot as plt
import numpy as np
import urllib.request
from keras.utils import normalize
import os
import glob
import cv2
from PIL import Image
from io import BytesIO
import requests
import uuid
import boto3
#from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, S3_BUCKET_NAME
import os
from dotenv import load_dotenv
from filehandler import upload_file_to_s3,get_presigned_file_url,delete_folder
from joblib import load
import io
load_dotenv()

AWS_ACCESS_KEY=os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY=os.getenv('AWS_SECRET_KEY')
S3_BUCKET_NAME=os.getenv('S3_BUCKET_NAME')
AWS_REGION=os.getenv('AWS_REGION')

s3=boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,

)


IMG_HEIGHT = 224
IMG_WIDTH  = 224
IMG_CHANNELS = 3
n_classes=3
################################################################
def multi_unet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #model.summary()

    return model


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_HEIGHT, IMG_CHANNELS=3)

model = get_model()

model.load_weights('model_unet\complete_modelwt.weights.h5')

class_colors = {
    0: [0, 0, 255],  # Red for class 0
    1: [0, 255, 0],  # Green for class 1
    2: [255, 0, 0]   # Blue for class 2
}
#####################################################
def read_image_from_url(url):
    # Download the image from the URL
    # response = urllib.request.urlopen(url)
    # image_data = response.read()
    # image_array = np.frombuffer(image_data, np.uint8)
    
    # # Decode the image using OpenCV
    # image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # img = cv2.resize(image, (IMG_HEIGHT, IMG_HEIGHT))

    image = cv2.imdecode(np.frombuffer(requests.get(url).content, np.uint8), cv2.IMREAD_COLOR)
    
    
    image=cv2.resize(image, (224, 224))
    #print(image)
    
    
    
    return image

def segment(urls):
    imgs=[]
    
    for url in urls:
        img=read_image_from_url(url)
        imgs.append(img)
    imgs=np.array(imgs)
    #print(imgs[0,:,:])
    #print(imgs.shape)
    segments=model.predict(imgs)
    segments=np.argmax(segments,axis=3)
    #print(segments.shape)
    
    #print(segments[0,:,:])
    count_0 = np.count_nonzero(segments[:,:,:] == 0)
    count_1 = np.count_nonzero(segments[:,:,:] == 1)
    count_2 = np.count_nonzero(segments[:,:,:] == 2)
    

    severity=float(count_1)/float(count_1+count_2)
    #print(count_1," ",count_2," ",severity)
    
    # color_image = np.zeros((segments[0,:,:].shape[0], segments[0,:,:].shape[1], 3), dtype=np.uint8)
    # for i in range(segments[0,:,:].shape[0]):
    #     for j in range(segments[0,:,:].shape[1]):
    #         color_image[i, j] = class_colors[segments[0,:,:][i, j]]
    # cv2.imwrite("segmented_image1.png", color_image)

    if severity==0:
        return 0
    elif severity<0.0420:
        return 1
    elif severity>=0.0420 and severity<0.0830:
        return 2
    elif severity>=0.0830 and severity<0.1380:
        return 3
    else:
        return 4
      

    # # print("Number of 0s:", count_0)
    # # print("Number of 1s:", count_1)
    # # print("Number of 2s:", count_2)
    



    



