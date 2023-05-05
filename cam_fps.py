"""
Test SSD dengan mendapatkan FPS


"""

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 

### Mengimport library
# For tensor
from array import array
from multiprocessing.resource_sharer import stop
from pyexpat.model import XML_CQUANT_REP
# from termios import CR2, CR3
from tokenize import cookie_re

from turtle import delay
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# For Load Model
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# For Detection
import cv2 
import numpy as np
import serial
import time

### Konfigurasi model
CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
# config


### Loading Model from Check Point

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-12')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

### Real-time Detection

category_index = label_map_util.create_category_index_from_labelmap('Tensorflow/workspace/annotations/label_map.pbtxt')

# Setup capture

cap = cv2.VideoCapture(0)

## Menampilkan fps kamera 

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

 
while True: 
    # global red_x
    ret, frame = cap.read()
    image_np = np.array(frame)

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = cv2.cvtColor(image_np.copy(), cv2.COLOR_BGR2RGB)

    """
    FPS 
    """

    # if video finished or no Video Input
    if not ret:
        break
 
    # Our operations on the frame come here
    gray = frame
 
    # resizing the frame size according to our need
    gray = cv2.resize(gray, (500, 300))
 
    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
    cv2.putText(image_np_with_detections, fps, (580, 50), font, 1, (0, 150, 0), 3, cv2.LINE_AA)

    """
    FPS
    """

    # Deteksi objek menjadi nilai tugas
    # This is the way I'm getting my coordinates
    boxes = detections['detection_boxes']

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)

    
    
    
    # This is the way I'm getting my coordinates
    boxes = detections['detection_boxes']
    # get all boxes from an array
    max_boxes_to_draw = boxes.shape[0]
    # get scores to get a threshold
    scores = detections['detection_scores']
    # this is set as a default but feel free to adjust it to your needs
    min_score_thresh=.5
    # # iterate over all objects found
    coordinates = []
    # class_id = int(detections['detection_classes'] + 1)

#pencet Q untuk close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

    cv2.imshow('object detection', image_np_with_detections)