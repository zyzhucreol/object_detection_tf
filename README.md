# TensorFlow object detection on a PC-based imaging system
Create and fine-tune an object detection model beyond those on the papers. The goal of this project is to physically deploy such a model on a PC with a camera and scientific imaging system.

# Pre-requisites:
* TensorFlow 2.5, cocoapi

# Models included in the tests
All models can be obtained from Tensorflow model zoo (link [4])
1. CenterNet HourGlass104 1024x1024
2. CenterNet MobileNetV2 FPN 512x512

# Useful resources
1. Installation instructions (especially helpful with the fix on pycocotool package: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation
2. Examples in the link above (using TF 2.X)
3. https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb
4. https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
