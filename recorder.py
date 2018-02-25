#######################################################
# Naive training set recorder
# @author: Christian Reichel
# Version: 0.1a
# -----------------------------------------------------
# This script makes a directory based on an entered
# name and a hash value and saves X images of the 
# user's face into that folder as training set for 
# object detection.
# For face detection, use Haar cascades to cut out
# just the face.
#
# Algorithm description:
# 1) Asks for a name
# 2) Makes new dir
# 3) Saves a bunch of images into that folder
#    _Advanced way: Saves keypoints into that folder.
# 
# Knowm challenges:
# [x] Record images
# [ ] Replace image recorder with keypoint recorder.
#######################################################

# IMPORTS
import os
import cv2 as cv
import time

def recorder():

    # Pretrained haar cascade classifiers.
    face_cascade = cv.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')

    # Ask for users name
    name = input("<recorder.py> How is your name?\n")

    # Make new dir if user not exists
    new_path_str = os.getcwd() + '/data/training/' + name + '_' + str(hash(int))
    print('<recorder.py> Hello ' + name + '. The algorithm will now learn to recognise your face. Please don\'t rotate your face. The recording start now.')
    time.sleep(0.100)
    print('-------------------------')
    print('<recorder.py> Make new folder at ' + new_path_str)
    os.makedirs(new_path_str, exist_ok=True)

    # Get camera image.
    camera = cv.VideoCapture(0)
    output, camera_image = camera.read()

    iterator = 0

    while output and iterator < 50:

        # Save grayscale image.
        camera_image_gray = cv.cvtColor(camera_image, cv.COLOR_BGR2GRAY)

        # Get face coordinates.
        faces = face_cascade.detectMultiScale(camera_image_gray, 1.3, 5)

        if len(faces) is 1:

            # Cut the face out of the camera image. 
            face_image = camera_image[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]

            # Write the image.
            filename = name + str(iterator) + '.png'
            print('<recorder.py> Save image: ' + filename)
            cv.imwrite(new_path_str + '/' + filename, face_image)

            # Iterate iterator.
            iterator += 1

            time.sleep(0.2)

        # Grab next camera image.
        output, camera_image = camera.read()

    # End procedures.
    print('<recorder.py> Ending process. ' + str(iterator) + ' images are created and saved into ' + new_path_str + '.')
    camera.release()
