import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# Defining paths and creating empty lists to store images
DATADIR = r'/home/gagandeep/Downloads/brain_tumor_dataset'
CATEGORIES = ['yes', 'no']
IMG_SIZE = 150
X = []
y = []

#lopping through the folders
for category in CATEGORIES:
	if category == 'yes': #label 1
		appendval = 1
	elif category == 'no': #label 2
		appendval = 0
	path=os.path.join(DATADIR, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #read image in grayscale
		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resizing the image
		new_array = new_array/255.0 # standardization
		X.append(new_array) # Append image data
		y.append(appendval) # Append label


# Reshape image - so that it can be used with CNN
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Spliting the data into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
	random_state=42, stratify=y)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

# Model Building
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=X.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25)

