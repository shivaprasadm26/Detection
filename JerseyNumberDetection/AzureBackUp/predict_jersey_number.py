# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:40:45 2020

@author: user
"""

import json
import cv2
import numpy as np
from yolo.frontend import create_yolo
from yolo.backend.utils.box import draw_scaled_boxes
import ntpath
import os
import yolo
from sklearn.cluster import AgglomerativeClustering

from imageai.Detection import ObjectDetection
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

detector = ObjectDetection()
custom = detector.CustomObjects(person=True)
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(yolo.PROJECT_ROOT, "yolo.h5"))
detector.loadModel()
print("detector loaded")

tf.logging.set_verbosity(tf.logging.ERROR)

cluster = AgglomerativeClustering(n_clusters=None, distance_threshold= 0.5, affinity='euclidean', linkage='ward')

DEFAULT_CONFIG_FILE = os.path.join(yolo.PROJECT_ROOT, "svhn", "config.json")
DEFAULT_WEIGHT_FILE = os.path.join(yolo.PROJECT_ROOT, "svhn", "weights.h5")
DEFAULT_THRESHOLD = 0.3
threshold = DEFAULT_THRESHOLD
config_file = DEFAULT_CONFIG_FILE 
weights_file = DEFAULT_WEIGHT_FILE 
with open(config_file) as config_buffer:
    config = json.loads(config_buffer.read())

# 2. create yolo instance & predict
yolo = create_yolo(config['model']['architecture'],
                   config['model']['labels'],
                   config['model']['input_size'],
                   config['model']['anchors'])
yolo.load_weights(weights_file)

def get_digits_boxes(img,show_img=False,save_img=False):
    img_path = img_path.replace('\\','/')
#    img_path = img_path.replace('\','/')
    image = cv2.imread(img_path)
    im_size = np.shape(image)
    boxes, probs = yolo.predict(image, float(threshold))
    labels = np.argmax(probs, axis=1) if len(probs) > 0 else [] 
    probs = np.max(probs,axis=1) if len(probs) > 0 else []

    # 4. save detection result
    image = draw_scaled_boxes(image, boxes, probs, config['model']['labels'])    
 
    centers = []
    
    for count in range(len(boxes)):
        
        box=boxes[count]
        
        x=(((box[0] + box[2]/2)/im_size[0]))
        y=(((box[1] + box[3]/2)/im_size[1]))
        centers.append([x,y])
    centers=np.array(centers)
    jersey_numbers = []

    if len(centers) > 1:
        cluster.fit_predict(centers)
        cluster_labels = cluster.labels_
        clusters = np.unique(cluster_labels)
        
        conf_score=[]
        for c_id in clusters:
            g_centers = centers[list(cluster_labels==c_id)]
            g_labels = labels[list(cluster_labels==c_id)]
            g_probs = probs[list(cluster_labels==c_id)]
            center_x = [center[0] for center in g_centers]
                
            g_labels=g_labels[np.argsort(center_x)]  
            number = int((''.join(str(label) for label in g_labels)))    
            jersey_numbers.append(number)
            conf_score.append(np.mean(g_probs))
    else:
        jersey_numbers = labels
        conf_score = np.mean(probs)
    numbers_string = ''  
    for number in jersey_numbers:
        numbers_string = numbers_string  + str(number) + ' '

    cv2.putText(image,'Jersey Number: %s' % numbers_string,(25,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        

#    labels=labels[np.argsort(center_x)]  
#    number = int((''.join(str(label) for label in labels)))    
    if save_img:
        
        output_path = os.path.join(ntpath.dirname(img_path) + '/' + (img_path.split("/")[-1]).split('.')[0] + '_out.jpg')
        cv2.imwrite(output_path, image)
    if show_img:
        cv2.imshow("jersey number detected image",image)           
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    print("{}-boxes are detected".format(len(boxes)))
    print("jersey numbers detected are {}".format(jersey_numbers))

    return jersey_numbers,conf_score,labels, boxes

def get_digits_boxes_img(image):
    im_size = np.shape(image)
    boxes, probs = yolo.predict(image, float(threshold))
    labels = np.argmax(probs, axis=1) if len(probs) > 0 else [] 
    probs = np.max(probs,axis=1) if len(probs) > 0 else []

    # 4. save detection result
    image = draw_scaled_boxes(image, boxes, probs, config['model']['labels'])    
 
    centers = []
    
    for count in range(len(boxes)):
        
        box=boxes[count]
        
        x=(((box[0] + box[2]/2)/im_size[0]))
        y=(((box[1] + box[3]/2)/im_size[1]))
        centers.append([x,y])
    centers=np.array(centers)
    jersey_numbers = []

    if len(centers) > 1:
        cluster.fit_predict(centers)
        cluster_labels = cluster.labels_
        clusters = np.unique(cluster_labels)
        
        conf_score=[]
        for c_id in clusters:
            g_centers = centers[list(cluster_labels==c_id)]
            g_labels = labels[list(cluster_labels==c_id)]
            g_probs = probs[list(cluster_labels==c_id)]
            center_x = [center[0] for center in g_centers]
                
            g_labels=g_labels[np.argsort(center_x)]  
            number = int((''.join(str(label) for label in g_labels)))    
            jersey_numbers.append(number)
            conf_score.append(np.mean(g_probs))
    else:
        jersey_numbers = list(labels)
        conf_score = [np.mean(probs)]
#     print("{}-boxes are detected".format(len(boxes)))
#     print("jersey numbers detected are {}".format(jersey_numbers))

    return jersey_numbers,conf_score



def detect_person_jersey_no(img_path):
    #print(img_path)
    img = cv2.imread(img_path)
    detections = detector.detectCustomObjectsFromImage(custom_objects=custom, input_image=img_path, output_image_path="out.jpg", minimum_percentage_probability=70, extract_detected_objects=False)
    
    jersey_numbers=conf_score=[]
    for det in detections:
        bb = det['box_points']
        cropped_im = img[bb[1]:bb[3],bb[0]:bb[2]]
        #cv2.imshow("cropped",cropped_im)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        out = get_digits_boxes_img(cropped_im)

        print(out)
        if len(out[0]):
            #print(out)
            jersey_numbers.extend(out[0])
            conf_score.extend(out[1])

    return jersey_numbers,conf_score
            


#img_path = 'D:\\Projects\\RetrieveByJerseyNumber\\data\\person_cropped_set_14_15_16_labelled\\images\\person (49).jpg'
#jersey_numbers,conf_score,labels, boxes= get_digits_boxes(img_path,show_img=False,save_img=False)


#import glob
#
#filenames= glob.glob("D:\Projects\RetrieveByJerseyNumber\data\person_cropped_set_14_15_16_labelled\images\*.jpg")
#for img_path in filenames:
#    jersey_numbers,conf_score,labels, boxes= get_digits_boxes(img_path,show_img=True,save_img=True)

#np.linalg.norm(np.array(centers[0])-np.array(centers[1]))




    


