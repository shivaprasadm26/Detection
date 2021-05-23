from imageai.Detection import ObjectDetection
import tensorflow as tf
import os
detector = ObjectDetection()
custom = detector.CustomObjects(person=True)
detector.setModelTypeAsYOLOv3()
detector.setModelPath("JerseyNumberDetection/yolo.h5")
print("detector loaded")

detector.loadModel()
img_path = r"C:\inetpub\wwwroot\wwwroot\images\TestNumber\person_cropped\person (4).jpg"
detections = detector.detectCustomObjectsFromImage(custom_objects=custom, input_image=img_path, minimum_percentage_probability=70,output_image_path="out.jpg", extract_detected_objects=False)
