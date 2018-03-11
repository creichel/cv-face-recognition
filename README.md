# CV: Face Recognition Application

Learns and identifies faces, with OpenCV and Python. Fully customizable and extendable.

## General approach

The goal is to have an extendable face recognition application, that is easy to use and to customize. The algorithm should learn continuously known faces and save additional training or validation sets for retraining the model and enhancing the accuracy.

## Clone / Install

1. Clone the git repository with the submodules. If you're working with the Terminal, here's the convenient code for you:

```
git clone --recursive https://github.com/n1g1r1/cv-face-recognition
```

2. Switch into cloned project folder and install necessary dependencies:

```
cd cv-face-recognition && pip install -r requirements.txt
```

3. Run the python script:
```
python main.py
```

.. or if you're that kind of person, here's a convenient one-liner:

```
git clone --recursive https://github.com/n1g1r1/cv-face-recognition && cd cv-face-recognition && pip install -r requirements.txt && python main.py
```

The application comes with a handy default `config.yml` file with default values. Check this file out and customize it if you want. You can also make multiple config files for your purposes and change the config file name in `main.py`.

## Usage

Call the script with

```
python main.py
```

You can customize the settings of the application by editing the `config.yml` file.

### The `config.yml`

```yaml
# General settings
general:
    face_classifier:    'haar' # Choose: lbp, haar
    recognizer:         'lbp' # Choose: lbp, eigen, fisher
    identifier_method:  'local' # Choose: local, remote
    capture_faces:      True
    train_model:        False
    classify:           True
    resize_factor:      0.5

# Training
training:
    set_path:         'data/training'
    number_of_images: 20
    resize:           False
    model_path:       'model'
    model_filename:   'LBPHData.xml'

# Validation
validation:
    make_set:         False
    set_path:         'data/validation'
    number_of_images: 20

# Test set
test:
    make_set:         True
    set_path:         'data/test'

# Server configuration for classification
server:
    url:              '' # Trailing slash
    classify:         ''
    train:            ''
    update:           ''
```

To detect faces in general, you can choose between two classifiers:

- `lbp` [Local Binarization Pattern Histogram ](https://www.docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms)
- `haar` [Haar Cascades](https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html)

To train and recognize the algorithm locally, you can choose between those three options in the `config.yml`:

- `lbp` [Local Binarization Pattern Histogram ](https://www.docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms)
- `eigen` [Eigenfaces](https://www.docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#eigenfaces)
- `fisher` [Fisherfaces](https://www.docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#fisherfaces)

You can even have your own remote recognizer that returns a JSON formatted label and confidence value. If so, fill in an `url` to the server, as well as the paths to the `classify`, `train` and `update` API interface in the `server` section in `config.yml`. For example, you might have the server address `http://recognize.faceidentifier.io`, so you might fill in the section like the following:

```yaml
# Server configuration for classification
server:
    url:              'http://recognize.faceidentifier.io/' # Trailing slash
    classify:         'classify'
    train:            'model/train'
    update:           'model/update'
```

You might want to also customize the `classifier_online` module to even match more your needs.

**More information:**

- online classifier module: https://github.com/n1g1r1/cv-module-face-classifier-online
- @quving's face classifier server API: https://github.com/Quving/cv-face-classifier
