from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from gtts import gTTS
from playsound import playsound
import numpy as np 
import cv2
import time
import os


model = load_model("face_rec.h5")
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

Capture = cv2.VideoCapture(0)

while True:
    # Capture frame from the Video Recordings.
    ret, frame = Capture.read()

    # Detect the faces using the classifier.
    faces = face_classifier.detectMultiScale(frame,1.3,5)

    if faces is ():
        face = None

    # Create a for loop to draw a face around the detected face.
    for (x,y,w,h) in faces:
        # Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        face = frame[y:y+h, x:x+w]

    # Check if the face embeddings is of type numpy array.
    if type(face) is np.ndarray:
        # Resize to (224, 224) and convert in to array to make it compatible with our model.
        face = cv2.resize(face,(224,224), interpolation=cv2.INTER_AREA)
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        # Make the Prediction.
        pred1 = model.predict(face)[0][0]
        pred2 = model.predict(face)[0][1]

        prediction = "Unknown"

        # Check if the prediction matches a pre-exsisting model.
        if pred1 > 0.9:
            print(pred1)
            prediction = "Bharath"

        elif pred2 > 0.9:
            print(pred2)
            prediction = "Other"

        else:
            prediction = "Unknown"

        label_position = (x, y)
        cv2.putText(frame, prediction, label_position, cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

    else:
        cv2.putText(frame, 'Access Denied', (20,60), cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if prediction == "Bharath":
            # time.sleep(3)
            tts = gTTS("Access Granted!")
            tts.save("audiofile.mp3")
            # playsound("audiofilex.mp3")
            os.system("audiofile.mp3")
            os.remove("audiofile.mp3")
            break
            video_capture.release()
            cv2.destroyAllWindows()

        else:
            tts = gTTS("Access Denied!")
            tts.save("audiofilex.mp3")
            # playsound("audiofilex.mp3")
            os.system("audiofilex.mp3")
            os.remove("audiofilex.mp3")

Capture.release()
cv2.destroyAllWindows()