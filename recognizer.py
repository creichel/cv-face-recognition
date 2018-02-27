#######################################################
# Face recognizer with SURF
# @author: Christian Reichel
# Based on: OpenCV tutorial for face recognition
# Version: 0.1
# -----------------------------------------------------
# This script detects and recognizes a face though
# SURF operation and recording of training data as a
# video file.
#
# A brief step by step explanation of the algorithm:
# 1) Camera image taken, SURF features computed.
# 2) Match against the own database of training data.
#    a) if match: continue
#    b) if not: 
#       1) Ask for a name.
#       2) Record images or a video as training data.
#       3) Match training data with current. If it
#          isn't matching, record again.
# 3) Show name as label.
# 
# Knowm challenges:
# [ ] Get it to work
#######################################################

# IMPORTS
import os
import numpy as np
import cv2 as cv
from modules.face_extractor import face_extractor
import requests

# Some parameters.
probability_threshold = 0.4
url = 'http://letsfaceit.quving.com'

# Get camera image.
camera = cv.VideoCapture(0)
output, camera_image = camera.read()

# Data JSON that will be sent to the server.
"""
Expected send JSON format:
{
    'mode' : 'detect', // Can be: train, validate, detect.
    'image' : camera_image // Image that should be analysed.
}
"""
data = {
    'mode' : 'detect', # detect, learn
    'image' : camera_image
}

# TODO: These should be the normal Python requests. Have to be tested.
# Might be that the response has to be casted into JSON format.
# response = requests.post(url, json=data)
# response_json = response.json()

# Dummy data just for testing the response. Remove when testing the 
# request.
response_dummy = {
    'json': {
        'label' : 'vinh',
        'probability' : '0.68'
    }
}

"""
Expected response JSON format:
{
    ... Additional information of POST request ...

    'json' : {
        'label' : label,
        'probability' : probability
    }
}
"""
label = response_dummy['json']['label']
probability = response_dummy['json']['probability']

# TODO: If probability is below a certain threshold, start learning procedure.
# NOTE: Maybe ask first if this is the matched person.
# if probability < probability_threshold:
#    face_extractor.build_training_set()
    # TODO: Send the data to server, activate learning algorithm.
    # TODO: Continue with algorithm.


print('Computer says: ' + str(label) + ' with a probability of ' + str(probability))

# TODO: When request works with one image, do it for every image. Code for 
# SURF detector is not needed but maybe neccesary for the training data, later.
#while output and cv.waitKey(1) == -1:    
    
    # Resize camera image for faster recognition.
    # Without blurring - we don't need undersampling handling at the moment.
    #camera_image = cv.resize(camera_image, (0,0), fx = 0.3, fy = 0.3)

    # Save grayscale image.
#    grayscale_image = cv.cvtColor(camera_image, cv.COLOR_BGR2GRAY)

    # Detect keypoints with SURF.
    ## detector = cv.xfeatures2d.SURF_create(4000)
    ## kp, des = detector.detectAndCompute(grayscale_image,None)

    # Draw the found keypoints onto the image.
    ## camera_image = cv.drawKeypoints(grayscale_image, kp, camera_image, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    # Show the result.
    # cv.imshow('Face Detector (with SURF)', camera_image)

    #data = {'image':camera_image}

    # response = requests.post(url, json=data)

    # PRINT OUT THE IMAGE LABEL HERE

    # Grab next camera image.
    # output, camera_image = camera.read()

# End procedures.
camera.release()
cv.destroyAllWindows()