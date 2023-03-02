import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2


# Load the trained model
model = load_model('Diabetic.h5')

# Create a dictionary to map the class indices to their names
class_names = {0: 'No DR', 1: 'DR'}

# Create a function to classify an image
def classify_image(image_path):

    # Load the image and process it with OpenCV
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    resized = cv2.resize(thresh, (224, 224))
    color = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    img_array = np.array(color)
    img_array = img_array / 255.0
    
    # Use the model to make a prediction
    predictions = model.predict(np.expand_dims(img_array, axis=0))
    # Get the class name of the predicted class
    class_idx = np.argmax(predictions[0])
    if predictions[0] > 0.6:
        class_idx = 1
    else:
        class_idx = 0
    print("Id = ",class_idx,"Predictions = ",predictions[0],"Classnames = ",class_names)
    class_name = class_names[class_idx]
    return class_name

# Create a function to open an image file dialog and display the selected image
def open_image_file():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if file_path:
        # Classify the selected image
        class_name = classify_image(file_path)
        print(file_path)
        # Update the GUI to display the selected image and its classification result
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        class_label.configure(text='Classification: ' + class_name)

# Create the GUI
root = tk.Tk()
root.title('DR Classification')
root.geometry('300x350')

# Create a button to open an image file dialog
button = tk.Button(root, text='Open Image', command=open_image_file)
button.pack(pady=10)

# Create a label to display the selected image
image_label = tk.Label(root)
image_label.pack()

# Create a label to display the classification result
class_label = tk.Label(root, text='Classification: ')
class_label.pack(pady=10)

root.mainloop()
