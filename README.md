# Snake_Detection
Snake Detection using Tensorflow Object Detection API

# Tensorflow Object Detection API

Creating accurate machine learning models capable of localizing and identifying multiple objects in a single image remains a core challenge in computer vision. The TensorFlow Object Detection API is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models.

# Requirements
1. Tensorflow 
2. Python 2.7 OR python 3.0
3. OpenCV 4.0

# Steps involved in training a model.
1. Collection of data : Based on the custom object to be ddetected, collect as more data as possible and in all possible variants.
2. Annotate/label the images, ideally with a program. I personally used "LabelImg". This process is basically drawing boxes around your object(s) in an image. The label program automatically will create an XML file that describes the object(s) in the pictures.
3. Split this data into train/test samples : Normally 10-15% of the total images will be of test samples.
4. Generate TF Records from these splits
5. Setup a .config file for the model of choice (you could train your own from scratch, but we'll be using transfer learning)
6. Train
7. Export graph from new trained model
8. Detect custom objects in real time!

Download Model from https://1drv.ms/u/s!AioA6iXbzJf_gQPXjGFV0tHNqolS

Note : Data(snake images) are provided based on request.
Warning : Model can not be used for commercial purpose.

