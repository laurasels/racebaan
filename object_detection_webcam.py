import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from overdrive import Overdrive
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from overdrive import Overdrive
from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation
# Any model exported using the `export_inference_graph.py` tool can be loaded
# here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the
# [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)
# for a list of other models that can be run out-of-the-box with varying speeds
# and accuracies.

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the
# object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90


def locationChangeCallback(addr, location, piece, speed, clockwise):
    # Print out addr, piece ID, location ID of the vehicle, this print
    # everytime when location changed
    print("Location from " + addr + " : " + "Piece=" + str(piece) +
          " Location=" + str(location) + " Clockwise=" + str(clockwise))
    print(piece)

    # Autos wisselen op hetzelfde moment
    """
    if piece == 18:
        print('change')
        car3.changeLaneRight(1000, 1000)
        car2.changeLaneRight(1000, 1000)
        #car3.changeSpeed(1000, 1000)
    if piece == 23:
        print('change')
        car3.changeLaneLeft(1000, 1000)
        car2.changeLaneLeft(1000, 1000)
    if piece == 40:
        print('change')
        car3.changeLaneLeft(1000, 1000)
        car2.changeLaneLeft(1000, 1000)
    """
    #if piece == 34:
     #   car3.changeSpeed(0, 1000)

# ## Download Model
def main(input_speed):

    if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
    	print ('Downloading the model')
    	opener = urllib.request.URLopener()
    	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    	tar_file = tarfile.open(MODEL_FILE)
    	for file in tar_file.getmembers():
    	  file_name = os.path.basename(file.name)
    	  if 'frozen_inference_graph.pb' in file_name:
    	    tar_file.extract(file, os.getcwd())
    	print('Download complete')
    else:
    	print('Model already exists')

    # ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading label map:
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `airplane`.  Here
    # we use internal utility functions, but anything that returns a dictionary
    # mapping integers to appropriate string labels would be fine

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Intializing the web camera device: 0 for internal and 1 for external camera
    # of the laptop used
    cap = cv2.VideoCapture(1)

    # Running the tensorflow session
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
       ret = True

       # Select which cars to use on the track using MAC address of the device
       #car2 = Overdrive("CD:5A:27:DC:41:89")
       car3 = Overdrive("DE:83:21:EB:1B:2E")
       car2 = Overdrive("FB:76:00:CB:82:63")
       #car3 = Overdrive("DB:DE:FF:52:CB:9E")

       # Set initial car speed and acceleration for the two cars
       initial_car_speed = input_speed
       initial_car_acceleration = 800
       car2.changeSpeed(initial_car_speed, initial_car_acceleration)
       car3.changeSpeed(initial_car_speed, initial_car_acceleration)

       while (ret):
          ret,image_np = cap.read()
          print(image_np)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          print(image_np_expanded)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
          classes_detected = classes[scores>0.5]
          if 13 in classes_detected:
              #print('detected')
              car3.changeSpeed(0, initial_car_acceleration)
              car2.changeSpeed(0, initial_car_acceleration)
              #car3.changeLaneRight(250, 250)
              car3.setLocationChangeCallback(locationChangeCallback)
              #print(car3.piece)
          elif 10 in classes_detected:
              print('verkeerslicht detected')
              #print('detected')
              car3.changeSpeed(0, initial_car_acceleration)
              car2.changeSpeed(0, initial_car_acceleration)
              #car3.changeLaneRight(250, 250)
              car3.setLocationChangeCallback(locationChangeCallback)
              """
          elif 1 in classes_detected:
              print('car detected')
              car3.changeSpeed(int(initial_car_speed/2), initial_car_acceleration)
              car2.changeSpeed(int(initial_car_speed/2), initial_car_acceleration)
              car3.setLocationChangeCallback(locationChangeCallback)
              """
          else:
              car3.changeSpeed(initial_car_speed, initial_car_acceleration)
              car2.changeSpeed(initial_car_speed, initial_car_acceleration)
              car3.setLocationChangeCallback(locationChangeCallback)
              #print(car3.piece)
              #drive(cv2)
          #print(image_np,boxes,classes,scores,category_index)
          #plt.figure(figsize=IMAGE_SIZE)
          #plt.imshow(image_np)
          cv2.imshow('image',cv2.resize(image_np,(1280,960)))
          if cv2.waitKey(25) & 0xFF == ord('q'):
              cv2.destroyAllWindows()
              cap.release()
              break


if __name__ == '__main__':
    input_speed = sys.argv[1]
    main(int(input_speed))
