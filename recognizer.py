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
import numpy as np
import cv2 as cv

# Get camera image.
camera = cv.VideoCapture(0)
output, camera_image = camera.read()

while output and cv.waitKey(1) == -1:    
    
    # Resize camera image for faster recognition.
    # Without blurring - we don't need undersampling handling at the moment.
    camera_image = cv.resize(camera_image, (0,0), fx = 0.3, fy = 0.3)

    # Save grayscale image.
    grayscale_image = cv.cvtColor(camera_image, cv.COLOR_BGR2GRAY)

    # Detect keypoints with SURF.
    sift = cv.xfeatures2d.SURF_create(4000)
    kp, des = sift.detectAndCompute(grayscale_image,None)

    # Draw the found keypoints onto the image.
    camera_image = cv.drawKeypoints(grayscale_image, kp, camera_image, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show the result.
    cv.imshow('Face Detector (with SIFT)', camera_image)

    # Grab next camera image.
    output, camera_image = camera.read()

# End procedures.
camera.release()
cv.destroyAllWindows()