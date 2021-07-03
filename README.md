# TensorFlow object detection on a PC-based imaging system
Create and fine-tune an object detection model beyond those on the papers. The goal of this project is to physically deploy such a model on a PC with a camera and scientific imaging system.

# Pre-requisites:
## Software infrastructure
* Visual Studio Community 2019
* Anaconda (latest version, installation for Everyone (requires Administrator)); Open Anaconda PowerShell to create a dedicated virtual environment (Do NOT run as Administrator): `conda create -n your_env_name --clone = base`
* Latest release of Protocol Buffers from https://github.com/protocolbuffers/protobuf/releases/tag/v3.17.3. Add the path to the binary exectuable protoc.exe to the user environmental variable PATH.
## Object-detection-specific packages
* opencv: This is required to import cv2 in Python. In virtual environment, do `conda install opencv-python`
* cocoapi: In virtual environment, compile from source link[5] with `pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI` (The compilation depends on Visual Studio Community 2019; An error converning setup.py will occur if Visual Studio is not installed)
## Installation instructions
* Activate the virtual environment with `conda activate your_env_name`;Create a new directory your_directory. All the following instructions must be performed within the virtual environment.
* Inside your_directory, clone TensorFlow pre-built models `git clone https://github.com/tensorflow/models.git`
*  Execute the following command in PowerShell to build .proto files
```
# From within your_directory/models/research/
Get-ChildItem object_detection/protos/*.proto | foreach {protoc "object_detection/protos/$($_.Name)" --python_out=.}
```
* Install TensorFlow pre-built models as python loadable modules
```
# From within your_directory/models/research/
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

# Models included in the tests
All models can be obtained from Tensorflow model zoo (link [4]). Usage examples in the file colab_webcam.py, mobilenet_webcam.py, resnet50_webcam.py
1. CenterNet HourGlass104 1024x1024 (197ms)
2. CenterNet MobileNetV2 FPN 512x512 (6ms)
3. CenterNet Resnet50 V2 512x512 (27ms)

# Useful resources
1. Instructions on fixing the errors with pycocotool package: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation
2. Examples in the link above (using TF 2.X)
3. https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb
4. https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
5. https://github.com/philferriere/cocoapi.git
