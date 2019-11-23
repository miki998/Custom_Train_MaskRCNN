import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

import matplotlib.pyplot as plt

# Keras shit
from keras.preprocessing.image import load_img,img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances

# define 81 classes that the coco model knowns about
class_names = ['bread-wholemeal', 'potatoes-steamed', 'broccoli', 'butter', 
            'hard-cheese', 'water', 'banana', 'wine-white', 'bread-white', 
            'apple', 'pizza-margherita-baked', 'salad-leaf-salad-green', 
            'zucchini', 'water-mineral', 'coffee-with-caffeine', 'avocado', 
            'tomato', 'dark-chocolate', 'white-coffee-with-caffeine', 'egg', 
            'mixed-salad-chopped-without-sauce', 'sweet-pepper', 'mixed-vegetables', 
            'mayonnaise', 'rice', 'chips-french-fries', 'carrot', 'tomato-sauce', 
            'cucumber', 'wine-red', 'cheese', 'strawberries', 'espresso-with-caffeine', 
            'tea', 'chicken', 'jam', 'leaf-spinach', 'pasta-spaghetti', 'french-beans', 'bread-whole-wheat']
 
# define the test configuration
class TestConfig(Config):
    NAME = 'food'

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 40  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.1
 	
def main():
    array = sys.argv[1:]

    if os.path.exists(array[0]): 
        path_to_weight = array[0]
        sys.exit(0)
    else: 
        print('path to weight does not exist')
        sys.exit(0)
    if os.path .exists(array[1]): path_to_image =array[1]
    else: 
        print('path to image does not exist')
        sys.exit(0)
    if float(array[2]) <= 1 and float(array[2]) >= 0: conf=array[2]
    else: 
        print('confidence must be a float')
        sys.exit(0)

    config = TestConfig() 
    config.DETECTION_MIN_CONFIDENCE = conf

	# define the model
	rcnn = MaskRCNN(mode='inference', model_dir='./load_weights', config=config)
	# load coco model weights
	rcnn.load_weights(path_to_weight, by_name=True)
	# load photograph
	img = load_img(path_to_image)
	img = img_to_array(img)
	# make prediction
	results = rcnn.detect([img], verbose=1)
	# get dictionary for first prediction
	r = results[0]
	# show photo with bounding boxes, masks, class labels and scores
	display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

if __name__ == '__main__':
	main()

