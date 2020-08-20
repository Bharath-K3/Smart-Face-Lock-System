"""
Importing the opencv module for computer vision and capturing images.
Importing os module to access the local system.
We are to rescale all our images captured from the default frame size
    to (224, 224) pixels because that is best if we want to try out
    transfer learning models like VGG16. We have already captured the
    captures in a RGB format. Thus we already have 3 channels and we 
    do not need to specify that. The required number of channels for a
    VGG-16 architecture is ideally (224, 224, 3).

"""

import cv2
import os

directory = "Bharath/"
path = os.listdir(directory)

for i in path:
    img_path = directory + i
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    cv2.imwrite(img_path, image)