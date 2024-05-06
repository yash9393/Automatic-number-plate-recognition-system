# Set the path to Tesseract-OCR executable in Google Colab
#pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"




import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time
from matplotlib import pyplot as plt

# Install pytesseract
!sudo apt install tesseract-ocr
!pip install pytesseract

# Importing Google Colab files module
from google.colab import files

# Specify the path to the uploaded image
image_path = '/content/image.jpg'

try:
    # Read the uploaded image
    image = cv2.imread(image_path)
    
    # Check if the image was successfully read
    if image is None:
        raise Exception("Unable to read the image. Please check the image path.")
    
    # Resize the image
    image = imutils.resize(image, width=500)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Perform edge detection
    edged = cv2.Canny(gray, 170, 200)

    # Find contours in the edged image
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    NumberPlateCnt = None

    # Loop over contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # If the approximated contour has four points, then assume that screen is found
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

    # Mask the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(image, image, mask=mask)

    # Configuration for tesseract
    config = ('-l eng --oem 1 --psm 3')

    # Run tesseract OCR on image
    text = pytesseract.image_to_string(new_image, config=config)

    # Print recognized text
    print("Recognized Text:", text)

    # Display the image
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Final Image')
    plt.show()

except Exception as e:
    print("An error occurred:", e)
