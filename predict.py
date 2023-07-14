import tensorflow as tf
import numpy as np
import pickle as pkl
import cv2 as cv
from utils import*

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure each string is either "ODD"
# or "EVEN" (without the quotes) depending on whether the hexadecimal number in
# the image is odd or even. Take care not to make spelling or case mistakes. Make
# sure that the length of the list returned as output is the same as the number of
# filenames that were given as input. The judge may give unexpected results if this
# convention is not followed.

model = tf.keras.models.load_model("model.h5")

def decaptcha(filenames):
    labels = []  
    for file in filenames:
        img = cv.imread(file)
        images,li = process(img)
        
        if li != 4:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            images = img[9:140+9, 350:500]
        im = cv.resize(images, (64, 64))
        im = im / 255
        index = model.predict(im.reshape(1, 64, 64, 1), verbose=False)
        if(index[0][0]<0.5):
            labels.append("EVEN")
        else:  
            labels.append("ODD")
    
    return labels