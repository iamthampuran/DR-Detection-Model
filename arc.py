import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
from PIL import Image
import numpy as np
import os


# Load the CSV file with the labels
df = pd.read_csv('ODIR-5K/ODIR-5K/input.csv')
# Extract the labels for DR
y = df['D'].values

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
model.add(layers.Dense(1, activation='relu'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Load the image files
folder_path = 'ODIR-5K/ODIR-5K/Training Images'
image_files = os.listdir(folder_path)

k=0
simple = []
img_array = []
for i in image_files:
    name = i[:i.find('_')]
    n1 = int(name)
    simple.append(n1)
simple.sort()

for i in simple:
    if k %2==0:
        j = str(i)
        j = j + "_right.jpg"
        k+=1
    # else:
    #     j = str(i)
    #     j = j + "_right.jpg"
    else:
        k+=1
        continue
    img_array.append(j)
    
image_files = img_array
# print("Image files = ",image_files)


X = []
for filename in image_files:
    # Open and resize the image with PIL
    image = Image.open(os.path.join(folder_path, filename)).resize((224, 224))
    print(filename)
    # Convert the PIL image to a numpy array
    img_array = np.array(image)
    # Normalize the pixel values to the range [0,1]
    img_array = img_array / 255.0
    # Add the image array to the list of X values
    X.append(img_array)
    # counter+=1

# Convert the list of X values to a numpy array
X = np.array(X)
print("Length = ",len(X))

# Train the model
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print('Accuracy:', accuracy)



