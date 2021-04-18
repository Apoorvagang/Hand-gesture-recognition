
import warnings
warnings.filterwarnings('ignore')

import keras
import matplotlib.pyplot as plt # for plotting
import os # provides a way of using operating system dependent functionality
import cv2 #Image handling library
import numpy as np
# shuffle the input data
import random
# Import of keras model and hidden layers for our convolutional network
from keras.layers import Conv2D, Activation, MaxPool2D, Dense, Flatten, Dropout
# splitting the input_data to train and test data
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import model_from_yaml
from keras.models import model_from_json

from sklearn.metrics import confusion_matrix
import seaborn as sn


CATEGORIES = ["01_palm", '02_l','03_fist','04_fist_moved','05_thumb','06_index','07_ok','08_palm_moved','09_c','10_down']
IMG_SIZE = 50
data_path = "leapGestRecog"

# Loading the images and their class(0 - 9)
image_data = []
for dr in os.listdir(data_path):
    for category in CATEGORIES:
        class_index = CATEGORIES.index(category)
        path = os.path.join(data_path, dr, category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                image_data.append([cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)), class_index])
            except Exception as e:
                pass

print(image_data[0])


random.shuffle(image_data)


input_data = []
label = []
for X, y in image_data:
    input_data.append(X)
    label.append(y)



plt.figure(1, figsize=(10,10))
for i in range(1, 10):
    plt.subplot(3, 3,i)
    plt.imshow(image_data[i][0], cmap='hot')
    plt.xticks([])
    plt.yticks([])
    plt.title(CATEGORIES[label[i]][3:])
plt.show()

#data normalization
input_data = np.array(input_data)
label = np.array(label)
input_data = input_data/255.0
input_data.shape

label = tf.keras.utils.to_categorical(label, num_classes=10, dtype='i1')

print(label)

# reshaping the data
input_data.shape = (-1, IMG_SIZE, IMG_SIZE, 1)

X_train, X_test, y_train, y_test = train_test_split(input_data, label, test_size=0.3, random_state=0)

model = keras.models.Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(Activation('relu'))


model.add(Conv2D(filters=32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=7, batch_size=32, validation_data=(X_test, y_test))

model.summary()


test_loss, test_accuracy = model.evaluate(X_test, y_test)

print('Test accuracy: {:2.2f}%'.format(test_accuracy*100))


cat = [c[3:] for c in CATEGORIES]
plt.figure(figsize=(10, 10))
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
sn.heatmap(cm, annot=True, xticklabels=cat, yticklabels=cat)
plt.plot()


#saving the model
model.save("model.h5")
print("Saved model to disk")

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
print('json file saved')


