import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from PIL import Image
import numpy as np
import os
import cv2

# Load the CSV file with the labels
df = pd.read_csv('ODIR-5K/ODIR-5K/input.csv')
# Extract the labels for DR
y = df['D'].values
print("Y = ",y)
# Define the CNN model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Load the image files
folder_path = '0'
image_files = os.listdir(folder_path)

folder = '0'
# print(os.listdir(folder))

# k=0
# simple = []
# img_array = []
# for i in image_files:
#     name = i[:i.find('_')]
#     n1 = int(name)
#     simple.append(n1)
# simple.sort()

# for i in simple:
#     if k %2==0:
#         j = str(i)
#         j = j + "_left.jpg"
#         k+=1
#     # else:
#     #     j = str(i)
#     #     j = j + "_right.jpg"
#     else:
#         k+=1
#         continue
#     img_array.append(j)
    
# image_files = img_array
# print("Image files = ",image_files)

y = []
X = []
print("Going inside directory 0")
for filename in image_files:
    # Open the image with OpenCV
    img = cv2.imread(os.path.join(folder_path, filename))
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Otsu's thresholding to binarize the image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert the binary image to get the dark areas as foreground
    thresh = cv2.bitwise_not(thresh)
    # Resize the image with OpenCV
    resized = cv2.resize(thresh, (224, 224))
    #trying to convert back to original
    color = cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR)
    # Convert the OpenCV image to a numpy array
    img_array = np.array(color)
    # Normalize the pixel values to the range [0,1]
    img_array = img_array / 255.0
    # Add the image array to the list of X values
    X.append(img_array)
    y.append(0)
    #Printing Filename
    print(filename)

folder_path  = '1'
image_files = os.listdir(folder_path)
print("Going inside directory 1")

for filename in image_files:
    # Open the image with OpenCV
    img = cv2.imread(os.path.join(folder_path, filename))
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Otsu's thresholding to binarize the image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert the binary image to get the dark areas as foreground
    thresh = cv2.bitwise_not(thresh)
    # Resize the image with OpenCV
    resized = cv2.resize(thresh, (224, 224))
    #trying to convert back to original
    color = cv2.cvtColor(resized,cv2.COLOR_GRAY2BGR)
    # Convert the OpenCV image to a numpy array
    img_array = np.array(color)
    # Normalize the pixel values to the range [0,1]
    img_array = img_array / 255.0
    # Add the image array to the list of X values
    X.append(img_array)
    y.append(1)
    #Printing Filename
    print(filename)

print("Size of x = ",len(X),"\nSize of y = ",len(y))

# Convert the list of X values to a numpy array
X = np.array(X)
y = np.array(y)
print("Length = ",len(X))

# Train the model
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy:', accuracy)

model.save('Diabetic.h5')


