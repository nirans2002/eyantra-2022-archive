import numpy as np
import cv2
from cv2 import aruco
import math
from pyzbar import pyzbar


# img_dir_path = "public_test_cases/"
# marker = 'qr'

# img_file_path = 'public_test_cases/qr_0.png'

# # read qr from image file and return the data
# def read_Qr_image(img_file_path):
#     img = cv2.imread(img_file_path)
#     Qr_codes_details = {}
#     Qr_codes = pyzbar.decode(img)
#     for Qr_code in Qr_codes:
#         (x, y, w, h) = Qr_code.rect
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         Qr_codes_details[Qr_code.data.decode('utf-8')] = (x + w / 2, y + h / 2)
#     return Qr_codes_details


# # ////////

# data = read_Qr_image(img_file_path)

# print(data)




img_file_path = 'public_test_cases/aruco_1.png'

def detect_ArUco_details(image):

    """
    Purpose:
    ---
    This function takes the image as an argument and returns a dictionary such
    that the id of the ArUco marker is the key and a list of details of the marker
    is the value for each item in the dictionary. The list of details include the following
    parameters as the items in the given order
        [center co-ordinates, angle from the vertical, list of corner co-ordinates] 
    This order should be strictly maintained in the output

    Input Arguments:
    ---
    `image` :	[ numpy array ]
            numpy array of image returned by cv2 library
    Returns:
    ---
    `ArUco_details_dict` : { dictionary }
            dictionary containing the details regarding the ArUco marker
    
    Example call:
    ---
    ArUco_details_dict = detect_ArUco_details(image)
    """    
    ArUco_details_dict = {} #should be sorted in ascending order of ids
    ArUco_corners = {}

    ##############	ADD YOUR CODE HERE	##############
        ##############	ADD YOUR CODE HERE	##############
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
    parameters = aruco.DetectorParameters_create()
    (corners, ids,rejected) =  aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    ids = ids.flatten()
    # print(ids)

    for (corner,id) in zip(corners,ids):
        corners = corner.reshape((4,2))
        (topLeft,topRight,bottomRight,bottomLeft) = corners

        topRight = (int(topRight[0]),int(topRight[1]))
        topLeft = (int(topLeft[0]),int(topLeft[1]))
        bottomRight = (int(bottomRight[0]),int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]),int(bottomLeft[1]))

        cornerCords = [topLeft,topRight,bottomLeft,bottomRight]


        center_x = int((topLeft[0]+ bottomRight[0])/2)
        center_y = int((topLeft[1]+ bottomRight[1])/2)

        centerCords = [center_x,center_y]

        # angle
		
        angle = (math.atan2(((topRight[0])-(topLeft[0])),((topRight[1]-topLeft[1]))))
		# print((round(math.degrees(angle)))%360)
        if angle<0:
            angle = (round(math.degrees(angle)))+360
        else:
            angle = round(math.degrees(angle))



        data = [centerCords,angle,cornerCords]

        ArUco_details_dict[id] = data
        ArUco_corners[id] = cornerCords       

    return ArUco_details_dict, ArUco_corners 


img = cv2.imread(img_file_path)
detect_ArUco_details(img)