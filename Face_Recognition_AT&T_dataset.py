
import os
import cv2
import sys
import pandas as pd
import numpy as np
import copy
from fnmatch import fnmatch 



### finding training images

def find_train_images(images_data):

    if images_data == "existing_dataset":
        folder='/AT&Ttrain/'
        patternmatch="*.pgm"

    if images_data== "real_time":
        folder='/AT&Ttest/'
        patternmatch="*.jpg"

    in_path = os.path.realpath("Face_Recognition_AT&T_dataset.ipynb")
    path = os.path.dirname(in_path) + folder

    

    Path_to_images=[]
    for dirname, subdirname, files in os.walk(path):
        for name in files :
            if fnmatch(name, patternmatch):
                Path_to_images.append(os.path.join(dirname,name))

    train_images=[]
    sizes=[]
    for path_of_image in Path_to_images:
        img_2D = cv2.imread(path_of_image, cv2.COLOR_BGR2GRAY)  
        img_1D = np.array(img_2D, dtype='float64').flatten() 
        train_images.append(img_1D)
        sizes.append(img_1D.size)

    size_of_image = train_images[0].size

    train_set_size = len(train_images)

    images_1d=np.array(train_images)

    images_1d=images_1d.T

    return images_1d, size_of_image, train_set_size, Path_to_images


## finding mean vector

def mean1d_(images_1d):
    mean1d=images_1d.mean(axis=1)
    return mean1d


## subtracting mean from image vectors

def images_mean_(images_1d,mean1d,train_set_size):
    images_mean=np.zeros([10,112*92],dtype='float64')
    images_mean=copy.deepcopy(images_1d)
    for i in range(0,train_set_size):
        images_mean[:,i]=images_1d[:,i]-mean1d
    return images_mean


## PCA algorithm
def PCA_(images_mean,k):
    CovarianceMatrix=np.matrix(images_mean.transpose()) * np.matrix(images_mean) 
    eigenvalues,eigenvectors = np.linalg.eig(CovarianceMatrix)

    indices_sorted=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[indices_sorted]
    eigenvectors=eigenvectors[indices_sorted]

    eigenvalues = eigenvalues[0:k]
    eigenvectors = eigenvectors[0:k,:]

    eigenvectors = eigenvectors.transpose()
    eigenvectors = images_mean * eigenvectors

    norms = np.linalg.norm(eigenvectors,axis=0)
    eigenvectors = eigenvectors/norms

    weights= eigenvectors.transpose()*images_mean

    return eigenvectors, weights


### Train
def train(dataset_name):

    images_1d, size_of_image, train_set_size, Path_to_images=find_train_images(dataset_name)

    mean1d=mean1d_(images_1d)
    images_mean=images_mean_(images_1d,mean1d, train_set_size)

    k = ((images_mean.shape[1])/2)+1

    eigenvectors,weights=PCA_(images_mean,k)


    print "size of images" , size_of_image

    return images_1d, mean1d, eigenvectors, weights, Path_to_images


## Test
def test(mean1d, eigenvectors, weights, test_image, Threshold=10000):
    
    test_image = np.array(test_image, dtype='float64').flatten()

    test_image_mean = test_image - mean1d
    
    
    test_image_mean = np.matrix(test_image_mean)
    test_image_mean=test_image_mean.transpose()
    weights_test= eigenvectors.transpose() * test_image_mean
    difference = weights - weights_test
    distance_faces = np.linalg.norm(difference,axis=0)
    
    closest_face_index = np.argmin(distance_faces)
    closest_face_distance = min(distance_faces) 

    if closest_face_distance > Threshold :
        closest_face_distance = -1

    return closest_face_index, closest_face_distance


## to find test images

def test_existing(mean1d, eigenvectors, weights,dataset_name, Threshold=10000):
    if dataset_name == "existing_dataset":
        folder='/AT&Ttest/'
        patternmatch="*.pgm"

    in_path = os.path.realpath("Face_Recognition_AT&T_dataset.ipynb")
    path = os.path.dirname(in_path) + folder


    Path_to_test_images=[]
    for dirname, subdirname, files in os.walk(path):

        for name in files :
            if fnmatch(name, patternmatch):
                Path_to_test_images.append(os.path.join(dirname,name))

    closest_face_index_lst=[]
    closest_face_distance_lst =[]

    for test_image_path in Path_to_test_images:
        img_2D = cv2.imread(test_image_path, cv2.COLOR_BGR2GRAY)  
        closest_face_index, closest_face_distance = test(mean1d, eigenvectors, weights, img_2D)

        closest_face_index_lst.append(closest_face_index)
        closest_face_distance_lst.append(closest_face_distance)

    return Path_to_test_images, closest_face_index_lst, closest_face_distance_lst



def print_help():
    print(""" FACE RECOGNITION 

    Options : 
        1.  Train and test AT&T dataset 
        
    

    Please enter the option (1) \n""")




def main1():
    
    print_help()

    command = raw_input()

    if command == '1':
        
    ## Train
        print('Training existing AT&T dataset\n')
        dataset_name = 'existing_dataset'
        images_1d, mean1dA, eigenvectors, weights, Path_to_images = train(dataset_name)
        

    ## Test
        Path_to_test_images, closest_face_index_lst, closest_face_distance_lst=test_existing(mean1dA, eigenvectors, weights,dataset_name)
        print("Recognising test images ")
          
        list_of_images = copy.deepcopy(Path_to_images)
        for idx in range(0,len(closest_face_index_lst)): 
            print("\nTest face at path %s"%(Path_to_test_images[idx]))
            name_of_face_class = os.path.basename(os.path.dirname(list_of_images[closest_face_index_lst[idx]]))
            print "This test face belongs to class %s"%(name_of_face_class)

        print("Done")

    else:
        print_help()
        main1()
        
        

def main():
    main1()


if __name__ == '__main__':
    main()




