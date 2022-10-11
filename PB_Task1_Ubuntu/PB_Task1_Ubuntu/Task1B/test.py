import numpy as np
import cv2
from cv2 import aruco
import math
from pyzbar import pyzbar


img_dir_path = "public_test_cases/"
marker = 'qr'

img_file_path = 'public_test_cases/qr_0.png'

# read qr from image file and return the data
def read_Qr_image(img_file_path):
    img = cv2.imread(img_file_path)
    Qr_codes_details = {}
    Qr_codes = pyzbar.decode(img)
    for Qr_code in Qr_codes:
        (x, y, w, h) = Qr_code.rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        Qr_codes_details[Qr_code.data.decode('utf-8')] = (x + w / 2, y + h / 2)
    return Qr_codes_details


# ////////

data = read_Qr_image(img_file_path)

print(data)