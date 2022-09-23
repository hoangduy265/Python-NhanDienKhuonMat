import tensorflow as tf

from unicodedata import name
from keras.models import load_model

from PIL import Image, ImageOps
import numpy as np

import os
import wikipedia
from gtts import gTTS
import playsound

import cv2

from webdriver_manager.chrome import ChromeDriverManager

wikipedia.set_lang('vi')
language = 'vi'
path = ChromeDriverManager().install()


def speak(text):
    tts = gTTS(text = text, lang = language, slow = False)
    tts.save("sound.mp3")
    playsound.playsound("sound.mp3", True)
    os.remove("sound.mp3")


camera = cv2.VideoCapture(0)

def capture_image():
    ret,frame = camera.read()
    if ret == True :
        cv2.imwrite('test.jpg',frame)
        

def face_detaction():
    np.set_printoptions(suppress=True)
    # Load the model
    model = tf.keras.models.load_model('keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open('test.jpg')
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)

    name = ["Anh Nen", "Hoàng Duy"]
    index = 1
    max_value = -1
    for i in range(0, len(prediction[0])):
        if max_value < prediction[0][i]:
            max_value = prediction[0][i]
            index = i
    print("Result: ", name[index])
    print("Exactly: ", max_value)

    speak("Xin chào " + name[index] )

while True:
    capture_image()
    face_detaction()
