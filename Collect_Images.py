"""
Importing the opencv module for computer vision and capturing images.
Importing os module to access the local system.
We are going to on our default webcam and then proceed to capture the
    images of our faces which is required for the dataset.

"""

import cv2
import os

capture = cv2.VideoCapture(0)

directory = "Bharath/"
path = os.listdir(directory)

count = 0

while True:
    ret, frame = capture.read()
    cv2.imshow('Frame', frame)

    """
    Reference: http://www.asciitable.com/
    We will refer to the ascii table for the particular waitkeys.
    1. Click a picture when we press space button on the keyboard.
    2. Quit the program when q is pressed

    """

    key = cv2.waitKey(1)

    if key%256 == 32:
        img_path = directory + str(count) + ".jpeg"
        cv2.imwrite(img_path, frame)
        count += 1

    elif key%256 == 113: 
        break

capture.release()
cv2.destroyAllWindows()