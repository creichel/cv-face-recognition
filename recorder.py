#######################################################
# Naive training set recorder
# @author: Christian Reichel
# Version: 0.1a
# -----------------------------------------------------
# This script makes a directory based on an entered
# name and a hash value and saves X images into that
# folder as training set for object detection.
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
    # Ask for users name
    name = input("<recorder.py> How is your name?\n")

    # Make new dir if user not exists
    new_path_str = os.getcwd() + '/data/training/' + name + '_' + str(hash(int))
    print('<recorder.py> Hello ' + name + '. The algorithm will now learn to recognise your face. Please rotate it slowly to make the recognition more stable. The recording start now.')
    time.sleep(0.100)
    print('-------------------------')
    print('<recorder.py> Make new folder at ' + new_path_str)
    os.makedirs(new_path_str, exist_ok=True)

    # Get camera image.
    camera = cv.VideoCapture(0)
    output, camera_image = camera.read()

    iterator = 0

    while output and iterator < 50:

        time.sleep(0.5)
        
        # Resize camera image for faster recognition.
        # Without blurring - we don't need undersampling handling at the moment.
        camera_image = cv.resize(camera_image, (0,0), fx = 0.3, fy = 0.3)

        # Save grayscale image.
        grayscale_image = cv.cvtColor(camera_image, cv.COLOR_BGR2GRAY)

        # Write the image.
        filename = name + str(iterator) + '.png'
        print('<recorder.py> Save image: ' + filename)
        cv.imwrite(new_path_str + '/' + filename, camera_image)

        # Grab next camera image.
        output, camera_image = camera.read()

        # Iterate iterator.
        iterator += 1

    # End procedures.
    print('<recorder.py> Ending process. ' + str(iterator) + ' images are created and saved into ' + new_path_str + '.')
    camera.release()
recorder()
