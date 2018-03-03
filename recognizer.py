#######################################################
# Face recognizer
# @author: Christian Reichel
# Version: 0.1
# -----------------------------------------------------
# This script detects and recognizes a face though
# R-CNN with Keras, by @quving
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

# IMPORTS.
import os               # path finding.
import cv2 as cv        # webcam image.
import time             # sleep time.

import online_identifier as identifier
import faceUI as ui
from modules.face_extractor import face_extractor

# Settings.
path_to_trainingdata = os.getcwd() + '/data/training'
path_to_testdata = os.getcwd() + '/data/test'

probability_threshold = 0.4 # TODO: min probability threshold to distinguish known and unknown faces.
camera_image_scale = 0.5

# DEV parameters.
save_faces = True

# Get camera image.
camera = cv.VideoCapture(0)
output, camera_image = camera.read()


def recognize():

    print('<recognizer.py recognize()> Starting recognizer ___________________________________________')
    print('Saving test data as images: ' + str(save_faces))

    # Make folder if test faces should be saved.
    if save_faces:
        # Make new dir if user not exists.
        dirname = 'test'
        new_path_str = path_to_testdata + '/' + dirname + '_' + str(hash(int))
        os.makedirs(new_path_str, exist_ok=True)
        print('Images will be saved into this directory: ' + new_path_str)

    print('Image scale ratio: ' + str(camera_image_scale))

    print('Classifier: ' + face_extractor.get_classifier_path())

    # Load pretrained haar cascade classifiers.
    face_cascade = cv.CascadeClassifier(face_extractor.get_classifier_path())

    # Get camera image.
    camera = cv.VideoCapture(0)
    output, camera_image = camera.read()

    iterator = 0

    while output and cv.waitKey(1) == -1:

        # Resize camera image for faster recognition.
        # Without blurring - we don't need undersampling handling at the moment.
        camera_image = cv.resize(camera_image, (0,0), fx = camera_image_scale, fy = camera_image_scale)

        # Save grayscale image.
        camera_image_gray = cv.cvtColor(camera_image, cv.COLOR_BGR2GRAY)

        # Get face coordinates.
        faces = face_cascade.detectMultiScale(camera_image_gray, 1.3, 5)

        # Recognize every face in image.
        for (x,y,w,h) in faces:

            face_image = camera_image[y:y+h,x:x+w]

            # Get person and probability for captured image.
            person, probability = identifier.identify(face_image)
            print('Detected person: ' + person + '. Probability: ' + probability)

            # Save faces if needed.
            if save_faces:
                # Write the image.
                filename = person + str(iterator) + '.jpg'
                cv.imwrite(new_path_str + '/' + filename, face_image)

                iterator += 1

            # Draw data onto image.
            camera_image = ui.labeled_box(camera_image, person, 'probability: ' + probability, (x,y), (x+w, y+h))

            # Show the result.
            cv.imshow('Face recognizer', camera_image)

        # Grab next camera image.
        output, camera_image = camera.read()

    print('<recognizer.py recognize()> Ending recognizer ___________________________________________')

    # End procedures.
    camera.release()
    cv.destroyAllWindows()

# Method call. Uncomment for debugging.
recognize()
