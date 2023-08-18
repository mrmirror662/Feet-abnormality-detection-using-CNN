
import cv2
import os
import numpy as np
import tensorflow as tf
from keras import layers
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        #print(f'{filename} \n')
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

IMG_SIZE = 256

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE)
])
feature = 'Normalfeet'
images = load_images_from_folder(f"td3/{feature}")
dest = f'td4/{feature}'
counter = 0
for image in images:
    result = resize_and_rescale(image)
    cv2.imwrite(f'{dest}/{feature}{counter}.jpg',np.array(result, dtype = np.uint8 ))
    counter+=1
    print(f"count:{counter}")