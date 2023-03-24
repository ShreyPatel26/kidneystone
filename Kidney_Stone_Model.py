# Importing Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator

# Setting the Path
path = 'Dataset'

# Loading the images
train_data_dir = os.path.join(path, 'Train')
test_data_dir = os.path.join(path, 'Test')

# Image Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training the Model
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(150,150), batch_size=32, class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(150,150), batch_size=32, class_mode='binary', shuffle=False)

#creating the ML model
model = Sequential()

#Adding convolutional layers
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Adding a second convolutional layer
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the Model
# set the number of epochs
model.fit(train_generator, steps_per_epoch=32, epochs=3, validation_data=test_generator, validation_steps=32)

# Saving the Model
model.save('kidney_stone_model.h5')
