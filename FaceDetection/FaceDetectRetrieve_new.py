# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:38:06 2019

@author: user
"""

import os
import glob
from keras.models import load_model
import cv2
from yolo_face_detect import GetFaces
import numpy as np
import shutil
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from time import sleep
import pickle    
#def GetModel():
#    base_model = VGG16(include_top=False, weights='imagenet')
#
#    x = base_model.output
#    x = GlobalAveragePooling2D()(x)
#    encoder = Model(input=base_model.input, output=x)
#    return encoder


MODEL_ROOT_PATH="./"
model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5')
print("face model loaded")
#Model_encoder = GetModel()
#print("feature extraction model loaded")


def detectRetrieve(QueryPath,SearchSetParentPath,UserName="User", NumImages=10):
#    print("Cutoff Inside",CutOff)
#    print("Cutoff float",float(CutOff))
#    SearchSetParentPath= "D:/Projects/FaceRetrieval/0453-20191108T103754Z-001/0453/"
#    MODEL_ROOT_PATH = "./pretrain/"
#    model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5')
    global model_face
#    global Model_encoder
    SearchSetPath = glob.glob(SearchSetParentPath+ "/*.jpg", recursive = False) 
    SearchSetPath.extend(glob.glob(SearchSetParentPath+ "/*.png", recursive = False))
    SearchSetPath.extend(glob.glob(SearchSetParentPath+ "/*.bmp", recursive = False))
    SearchSetPath.extend(glob.glob(SearchSetParentPath+ "/*.tiff", recursive = False))
    SearchSetPath.extend(glob.glob(SearchSetParentPath+ "/*.tif", recursive = False))
    SearchSetPath.extend(glob.glob(SearchSetParentPath+ "/*.JPEG", recursive = False))
#    print("search images@#####",SearchSetPath)
    encoding_file = os.path.join(SearchSetParentPath,"searchset_encodings.p")
    if(os.path.exists(encoding_file)):
        print("loading existing encodings")
        SearchFaceEncodings = pickle.load(open(encoding_file, "rb"))
        print("Encodings are loaded successfully")
    else:    
        print("Encodings not found!\nCreating encodings for database images")
        SearchFaceEncodings =[]
        count = 1
        for im in SearchSetPath:
            file_name = im.split("\\")[-1]
            img = cv2.imread(im)
            if img is None:
                print("Could not read input image")
                continue
            FaceEncodings = []
            FaceImages,imgg, fv = GetFaces(img, model_face)
            cv2.imshow("data",cv2.resize(imgg,None,fx=0.25,fy=0.25))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
            ImDetails = {}
            ImDetails['name'] = file_name
            ImDetails['feature'] = fv
            SearchFaceEncodings.append(ImDetails)
    #        print("features",fv)
    #        if len(FaceImages) > 0:
    #            
    #            for face in FaceImages:
    #                
    #                image = cv2.resize(img, (224, 224))
    #                # convert the image pixels to a numpy array
    #                image = img_to_array(image)
    #                # reshape data for the model
    #                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #                # prepare the image for the VGG model
    #                image = preprocess_input(image)
    #        
    #                FaceEncoding = Model_encoder.predict(image)
    #                FaceEncodings.append(FaceEncoding) 
    #        SearchFaceEncodings.append(FaceEncodings)
            print("Processed Searchset image " + str(count))
            count +=1
        encoding_file = os.path.join(SearchSetParentPath,"searchset_encodings.p")
        pickle.dump(SearchFaceEncodings, open(encoding_file, "wb" ))
        print("Encodings created successfully")
#    QueryPath = "D:\\Projects\\FaceRetrieval\\0453-20191108T103754Z-001\\0453\\CC_0453_C2_008.JPG"
    
    
#    QueryPath.replace('\',"'/")
    #def RetrieveImage(QueryPath, SearchSetParentPath, DestPath, Model_encoder, model_face, CutOff, NumImages):
    QImages= glob.glob(QueryPath+ "/*.jpg", recursive = False) 
    QImages.extend(glob.glob(QueryPath+ "/*.png", recursive = False))
    QImages.extend(glob.glob(QueryPath+ "/*.bmp", recursive = False))
    QImages.extend(glob.glob(QueryPath+ "/*.tif", recursive = False))
    QImages.extend(glob.glob(QueryPath+ "/*.tiff", recursive = False))
    QImages.extend(glob.glob(QueryPath+ "/*.JPEG", recursive = False))
    QImage = cv2.imread(QImages[0])
    print("Query Image",QImages[0])
    
    FacesQuery, DetectedImage, fv = GetFaces(QImage, model_face)
    
    cv2.imshow("QueryImage",cv2.resize(DetectedImage,None,fx=0.25,fy=0.25))
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    
#    print("HHHHHHHHHHHH",len(FacesQuery))
    if len(FacesQuery) > 0 :
        QFace = FacesQuery[0]    
    else:
        print("no face detected in query!! try different image")
        exit()
        
#    image = cv2.resize(QFace, (224, 224))
#    # convert the image pixels to a numpy array
#    image = img_to_array(image)
#    # reshape data for the model
#    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#    # prepare the image for the VGG model
#    image = preprocess_input(image)
#    
#    QCode = Model_encoder.predict(image)
    QCode = fv[0]
    Scores = []
    # read input parent_path = "D:/Projects/FaceRetrieval/0453-20191108T103754Z-001/0453/"
    
    FinalSearchFaceEncodings= []
    distanceToSearchset = []
    SelectedImageNames = []
    count = 0
    for im in SearchFaceEncodings:
        file_name = im['name']
#        ImDetails= SearchFaceEncodings[count]
        FaceEncodings=im['feature']
        if len(FaceEncodings ) > 0:
            # initialize the distance list
            distances = []
    #        SearchFaceEncodings.append(FaceEncodings)
#            for code in FaceEncodings:
#        		# for each code of the training images, compute the Euclidean distance between the train image and test image
#                distance = np.linalg.norm(code - QCode)
#                distances.append(distance) # append to the distance list
            distances = [np.linalg.norm(code-QCode) for code in FaceEncodings]
            nb_elements = len(FaceEncodings) # get the totale number of images in the training set
            distances = np.array(distances) # convert the distance list to a numpy array
            FaceEncodings_index = np.arange(nb_elements) # creae an index list from 0 - nb_elements
            	
            # create a numpy stack with the distances, index_list
            distances_with_index = np.stack((distances, FaceEncodings_index), axis=-1)
            sorted_distance_with_index = distances_with_index[distances_with_index[:,0].argsort()] # sort the stack
            
            sorted_distances = sorted_distance_with_index[:, 0].astype('float32') # change the datatype
            sorted_indexes = sorted_distance_with_index[:, 1]
            kept_indexes = int(sorted_indexes[:1])
#            SelectedFace =  FaceImages[int(kept_indexes)] 
            
            FinalSearchFaceEncodings.append(FaceEncodings[kept_indexes])
    #        print(len(SearchFaceEncodings))
            SelectedImageNames.append(file_name) 
    # initialize the distance list
    distances = []
#    print(len(FinalSearchFaceEncodings))
    distances = [np.linalg.norm(code-QCode) for code in FinalSearchFaceEncodings]
#
#    for code in FinalSearchFaceEncodings:
#        # for each code of the training images, compute the Euclidean distance between the train image and test image
#        distance = np.linalg.norm(code - QCode)
#        distances.append(distance) # append to the distance list
    
    nb_elements = len(FinalSearchFaceEncodings) # get the totale number of images in the training set
    distances = np.array(distances) # convert the distance list to a numpy array
    trained_code_index = np.arange(nb_elements) # creae an index list from 0 - nb_elements
    	
    # create a numpy stack with the distances, index_list
    distances_with_index = np.stack((distances, trained_code_index), axis=-1)
#    print("dist index", distances_with_index)
    sorted_distance_with_index = distances_with_index[distances_with_index[:,0].argsort()] # sort the stack
    
    sorted_distances = sorted_distance_with_index[:, 0].astype('float32') # change the datatype
#    print(sorted_distances)
#    print("max distance %d min distance %d"%(np.max(sorted_distances),np.min(sorted_distances)))
    sorted_indexes = sorted_distance_with_index[:, 1]
#    print("sorted index",sorted_indexes)
    NumImages = int(NumImages)
    kept_indexes = sorted_indexes[:NumImages] # Get the first 25 indexes of the sorted_indexes list

#    print("cutoff",CutOff)
#    kept_indexes = sorted_indexes[list(map(lambda x: x <=CutOff ,sorted_distances))]
#    print(sorted_indexes)
#    print(kept_indexes)
    SelectedImageNames = np.array(SelectedImageNames)
    Kept_images = SelectedImageNames[kept_indexes.astype('int16')]    
    
    DestPath = os.path.join(QueryPath, UserName)# + '\\'
    print("Destination Path",DestPath)
    if os.path.exists(DestPath):
        shutil.rmtree(DestPath, ignore_errors=True)
    sleep(2)    
    os.mkdir(DestPath)
    rank=1
    for im in Kept_images:
#        print(type(im))
#        print(im)
        source = os.path.join(SearchSetParentPath, im)
#        print(source)
        
        destination = os.path.join(DestPath,str(rank)+"_"+im)
        rank+=1
        print(destination)
        dest = shutil.copyfile(source, destination) 
    print("Images are extracted in",DestPath)
    #    break
    
#if __name__=='__main__':
    
