# Snake_Detection
Snake Detection using Tensorflow Object Detection API

# Tensorflow Object Detection API

Creating accurate machine learning models capable of localizing and identifying multiple objects in a single image remains a core challenge in computer vision. The TensorFlow Object Detection API is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models.

# Requirements
1. Windows/Ubuntu
2. GPU (NVIDIA 1070 is used)
3. Tensorflow
4. Python v2.7 or Python > v3.0
5. OpenCV > v4.0

# Directory structure recommondation: 
![alt_text](https://github.com/SaiKrishnaTheGreat/Snake_Detection/blob/master/images/1.png)

# Stages involved in training a model :
1. Data Collection.
2. Annoation.
3. Train-Test Data.
4. Generate TF Records files. 
5. Model Architecture.
6. Train.
7. Frozen model generation.
8. Inference.

# 1. Data Collection :
a) Based on custom object, collect as more data as possible. 
b) Increase the data by applying image processing techniques like R,G,B pixel intensity increment, rotation.
c) Collect the objects in varient possible distances.

# 2. Annotation:
Annotation of images also called as Labeling data. “Labelimg” is the tool to be used for generation of annoation files. LabelImg is a graphical image annotation tool.It is written in Python and uses Qt for its graphical interface.Annotations are saved as XML files in PASCAL VOC format, the format used by ImageNet. Besdies, it also supports YOLO format.
Download tool from github repo. 
      I. Tool usage:
        a) Clone the repo.
        b) Run ‘python labelImg.py’
      II. label the image:
        a) Click 'Change default saved annotation folder' in Menu/File.
        b) Click 'Open Dir'.
        c) Click 'Create RectBox'.
        d) Click and release left mouse to select a region to annotate the rect box.
        e) You can use right mouse to drag the rect box to copy or move it.
           The annotation will be saved to the folder you specify. 
![alt_text](https://github.com/SaiKrishnaTheGreat/Snake_Detection/blob/master/images/2.jpg)

# 3. Train-Test Data:
Split the data into Train and Test data. Normally, 70% of data will be used for Training and 30% data will be used for Testing. 
# 4. Generate TFRecord files:
TFRecord files are Binary representation of the data. Convert the data from stage #3 to train.tfrecord and test.tfrecord files respectively.  
  I. XML to CSV conversion : Convert all XMLs to a csv file using below python script (Many techniques can be used). Run the script for train_data and test_data. 
![alt-text](https://github.com/SaiKrishnaTheGreat/Snake_Detection/blob/master/images/3.png)
  II. CSV to TFRecord file: Grab “generate_tfrecord.py” from tensorflow/models directory. And the only modification need to change is ‘class_text_to_int’ with the label assigned in stage #2.
  
# 5. Model Architecture:
  1. Selecting a model architecture can be a pretrained model or new architecture. As new architecture takes days of time to train, prefer pretrained model. Download weights from tensorflow zoo. Select a model. 
  2. pbtxt file: Object-class ID represented using object_class.pbtxt file.
 ![alt_text](https://github.com/SaiKrishnaTheGreat/Snake_Detection/blob/master/images/4.png)
  ID number increases based on the number of objects that to be trained. (Note: label name must be same which is used in stage #2.
  3. Configuration files: modify the configuration files for ‘num_classes’ and ‘PATH_TO_BE_CONFIGURED’ with the number of classes to be trained and train.tf record and test.tfrecord. (Note: total of 6 changes are important. Other changes depends on model, data, train/test examples).

# 6. Train:
  1. clone the tensorflow models repo.
  2. From models/research/object_detection, run python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=<config_file>.
  3. On successful run, 3 files will be generated .data, .index, .meta which together called as “Unfrozen Model”.
  4. Based on the configuration, unfrozen models are generated for various step_size.

# 7. Frozen Model Generation:
  1. From models/research/object_detection, run python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training/<configfile> --trained_checkpoint_prefix training/model.ckpt-<step_size> --output_directory <output_dir>.
  2. Result is frozen model “<output_name>.pb”.

# 8. Inference:
  1. Change the path for frozen model, object_detection.pbtxt, num_classes as reference data in inference script.
  2. From models/research/object_detection, run python3 inference_image.py for images and python3 inference_video.py for video/webcam.
Based on the input, get the detections on the frame with bounding boxes.

# Results
![alt text](https://github.com/SaiKrishnaTheGreat/Snake_Detection/blob/master/results/result1.jpeg)
![alt_text](https://github.com/SaiKrishnaTheGreat/Snake_Detection/blob/master/results/result2.png)
![alt text](https://github.com/SaiKrishnaTheGreat/Snake_Detection/blob/master/results/result3.png)

Download Model from https://1drv.ms/u/s!AioA6iXbzJf_gQPXjGFV0tHNqolS

Note : Data(snake images) are provided based on request.
Warning : Model can not be used for commercial purpose.

