# Import needed libraries.
import os as os         # File and directory handling.
import cv2 as cv        # Image operations.
import yaml as yaml     # Reading config file.
import time as time     # time measurement.

# Import custom modules.
from modules.detector import detector
from modules.trainer import trainer
from modules.classifier import classifier
from modules.classifier_online import classifier as classifier_online
import ui

# Config file setting. Default: 'config.yml'.
config_file = 'config.yml'





def recognize():

    print('=> Start face recognition application _______________________________')





    # Check if config file exists.
    if not os.path.isfile(config_file):
        print('No config file found. Please make sure you have a config file saved. Have a look into the documentation.')

    # Load config.
    config = yaml.safe_load(open(config_file))

    ## Apply settings from config file.
    # General settings.
    face_classifier     = config['general']['face_classifier']
    recognizer_method   = config['general']['recognizer']
    mode                = config['general']['identifier_method']
    capture_faces       = config['general']['capture_faces']
    train_model         = config['general']['train_model']
    classify            = config['general']['classify']
    resize_factor       = config['general']['resize_factor']

    # Training settings.
    training_set_path   = config['training']['set_path']
    no_of_trainingimg   = config['training']['number_of_images']
    resize_trainingimg  = config['training']['resize']
    model_path          = config['training']['model_path']
    model_filename      = config['training']['model_filename']

    # Validation settings.
    make_validation_set = config['validation']['make_set']
    validation_set_path = config['validation']['set_path']
    no_of_validationimg = config['validation']['number_of_images']

    # Test settings.
    make_test_set       = config['test']['make_set']
    test_set_path       = config['test']['set_path']

    # Server settings.
    server_url_classify = config['server']['url'] + config['server']['classify']
    server_url_train    = config['server']['url'] + config['server']['train']
    server_url_update   = config['server']['url'] + config['server']['update']





    ## Set recognizer.
    recognizer = None

    if recognizer_method == 'lbp':

        recognizer = cv.face.LBPHFaceRecognizer_create()

    elif recognizer_method == 'eigen':

        recognizer = cv.face.EigenFaceRecognizer_create()

    elif recognizer_method == 'fisher':

        recognizer = cv.face.FisherFaceRecognizer_create()

    # Recognizer is not needed if online is chosen or the user only wants to capture faces.
    elif identifier_method != 'remote' and (train_model or classify):

        print('Invalid recognizer method found. Please check your config file and choose:\n- lbp\n- eigen\n- fisher')
        return





    ## Capture face images from webcam if set.
    if capture_faces:

        print('Start capturing process...')

        # Ask for users name.
        label = input("What is your name?\n")

        ## Make folder.
        folder_hash = str(hash(time.time()))
        abs_training_set_path = training_set_path + '/' + label + '.' + folder_hash

        abs_validation_set_path = None
        if make_validation_set:
            abs_validation_set_path = validation_set_path + '/' + label + '.' + folder_hash

        # Make new dir if user not exists.
        print('Hello ' + label + '. The algorithm will now learn to recognise your face. The recording starts now.')

        # Wait time for readability for the user.
        time.sleep(0.200)

        print('_____________________________________________________________________')

        print('Make new folder at ' + abs_training_set_path)
        os.makedirs(abs_training_set_path, exist_ok = True)

        # Make new dir for validation set if needed.
        if make_validation_set:

            print('Make new folder at ' + abs_validation_set_path)
            os.makedirs(abs_validation_set_path, exist_ok = True)

        print('=> Start building training set...')

        # Start capturing process.
        trainer.capture(detector = detector, label = label, resize = resize_trainingimg, resize_factor = resize_factor, classifier = face_classifier, make_training_set = True, training_set_size = 20, make_validation_set = make_validation_set, validation_set_size = 20, training_set_path = abs_training_set_path, validation_set_path = abs_validation_set_path)

        # New training data, but don't train the model.
        if not train_model:

            # Ask if user really want to skip training model with new data.
            choice = input('Captured new data but value of training model is ' + str(train_model) + '. Are you sure you want to skip re-training the model? (y/N)')

            # Only accept valid input
            while choice != 'N' and choice != 'y':

                choice = input('Not allowed input. Are you sure you want to skip re-training the model? Enter y for yes or N for no. (y/N)')

            # Change train_model value to true to train the model afterwards.
            if choice == 'N':

                train_model = True

            else:

                print('Ok, will skip training the model.')





    ## Train the model if needed.
    model_file_path = model_path + '/' + model_filename

    # If there's no pretrained model, it can be loaded if wanted.
    if os.path.isfile(model_file_path) and not train_model:

        # Read the model.
        try:

            recognizer.read(model_file_path)
            print('Load trained classifier model from ' + model_file_path)

        except cv.error as e:

            print(e)

    # If there's no pretrained model or the user chooses to (re-)train the model, do so.
    else:

        # Make a specific output if user doesn't want to train the model but there's no pretrained model to use.
        if not train_model:

            print('No pretrained model found. Will train now.')

        # Start training.
        trainer.train(recognizer, training_set_path)

        # Save the trained model.
        os.makedirs(model_path, exist_ok = True)

        print('Saving trained classifier model as ' + model_path + '/' + model_filename)
        recognizer.write(model_file_path)





    ## Classify webcam image.
    if classify:

        print('Start classifying process...')

        # Get camera image.
        camera = cv.VideoCapture(0)
        output, camera_image = camera.read()

        while output and cv.waitKey(1) == -1:

            # Resize camera image for faster recognition.
            camera_image = cv.resize(camera_image, (0,0), fx = resize_factor, fy = resize_factor)

            # Save grayscale image.
            camera_image_gray = cv.cvtColor(camera_image, cv.COLOR_BGR2GRAY)

            # Start timer.
            detection_time = time.time()

            # Get face coordinates.
            faces, eyes, image = detector.detect(camera_image, classifier = face_classifier, resize = False, detect_eyes = False, draw_bounding_boxes = False)

            # Stop timer.
            detection_time = time.time() - detection_time

            # Recognize every face in image.
            for (x,y,w,h) in faces:

                face_image = camera_image[y:y+h,x:x+w]
                face_image_gray = camera_image_gray[y:y+h,x:x+w]

                # Start timer.
                class_time = time.time()

                # Get label and confidence for captured image.
                if mode == 'local':

                    label, confidence = classifier.classify(recognizer, face_image_gray)

                else:

                    label, confidence = classifier_online.classify(server_url_classify, face_image)

                # Stop timer.
                class_time = time.time() - class_time

                # Print results.
                print('Label: ' + str(label) + '. Confidence: ' + str(round(confidence, 2)) + '. Detection time: ' + str(round(detection_time, 2)) + 's. Class. time: ' + str(round(class_time, 2)) + 's.')

                # Draw data onto image.
                camera_image = ui.labeled_box(camera_image, str(label), 'conf.: ' + str(round(confidence,2)), (x,y), (x+w, y+h))

            # Show the result.
            cv.imshow('Face recognizer', camera_image)

            # Grab next camera image.
            output, camera_image = camera.read()

        # End procedures.
        camera.release()
        cv.destroyAllWindows()





    # End procedures.
    camera.release()
    cv.destroyAllWindows()

recognize()
