from random import shuffle
import cv2
import os
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        #print(f'{filename} \n')
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

x = []
y = []
tdata = 'tdata'
KK = 'KnockedKnees'
NK = 'NormalKnees'
FF = 'FlatFeet'
NF = 'NormalFeet'
np.random.seed(42) 
x_=  load_images_from_folder(tdata+'/'+KK);
for o in x_:
    y.append(np.array([0,1]))
x.extend(x_)

x1_=  load_images_from_folder(tdata+'/'+NK);
for o in x1_:
    y.append(np.array([1,0]))
x.extend(x1_)
for (k,h) in zip(x,y):
    print(np.shape(k))
    print(f'{h}\n')
np.random.shuffle(x)
np.random.shuffle(y)
batch_size = 32
img_height = 256
img_width = 256
train_data_dir = "tdata"
model = Sequential([
  layers.Rescaling(1/255, input_shape=(img_height, img_width, 3)),
  layers.RandomFlip("horizontal",input_shape=(img_height, img_width,3)),
  layers.RandomZoom(0.1),
  layers.RandomContrast(0.4),
  layers.RandomRotation(0.1),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(2,activation='softmax'),
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()
epochs = 25
history = model.fit(np.array(x), np.array(y) ,batch_size=batch_size, epochs=epochs, validation_split=0.1,shuffle=True)
model.save('modelkek')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()