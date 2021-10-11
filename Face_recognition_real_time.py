
import os
import cv2
import sys
import pandas as pd
import numpy as np
import copy
from fnmatch import fnmatch 

#### capturing images for making a dataset

def click_images(name):

    print("Take photos for training real-time dataset ")
    print("Press Enter to click photos. Take atleast 10 photos")

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("new_face")

    img_counter =0 

    while True: 
        ret, frame = cam.read()

        cv2.imshow("new_face", frame)

        if not ret : 
            break

        k = cv2.waitKey(1)

        if k%256 == 27:       # ESC for closing 
            print "Closing camera"
            break

        elif k%256 == 13:       # carriage return for clicking photos 

            image_name = "{}_frame_{}.jpg".format(name,img_counter)

            in_path = os.path.realpath("Face_recognition_real_time.py")
            folder = "/real_time/%s/"%(name)
            path = os.path.dirname(in_path) + folder

            try:
                os.makedirs(path)
                cv2.imwrite(path+ image_name, frame)
            except:
                cv2.imwrite(path+ image_name, frame)
                


            img_counter+=1
            print "photo %d clicked \n"%(img_counter)

    cam.release()

    cv2.destroyAllWindows()


### haarcascades for face detection

def findface_haarcascades(name):
    face_cascade = cv2.CascadeClassifier('/home/pranjali/.local/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    in_path = os.path.realpath("Face_recognition_real_time.py")
    folder = "/real_time/%s/"%(name)
    path = os.path.dirname(in_path) + folder

    patternmatch="*.jpg"

    Path_to_images=[]
    for dirname, subdirname, files in os.walk(path):
        #for files in subdirname:
            for names in files :
                if fnmatch(names, patternmatch):
                    Path_to_images.append(os.path.join(path,name))
                    image_path = path + names

                    img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
		    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		    #img = img[:,:,0]
                    faces= face_cascade.detectMultiScale(img,1.3,5)

                    for (x,y,w,h) in faces:
                        face_image = img[y:y+h,x:x+w]
                        face_image = cv2.resize(face_image,(40,40))
			#print "in haarcascades"
			#print face_image.shape
                        cv2.imwrite(image_path, face_image)


    print "Done extracting faces of %s\n"%(name)


### finding training images

def find_train_images(images_data):

    if images_data == "existing_dataset":
        folder='/faces/'
        patternmatch="*.pgm"

    if images_data== "real_time":
        folder='/real_time/'
        patternmatch="*.jpg"

    in_path = os.path.realpath("Face_recognition_real_time.py")
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
        img_1D = np.array(img_2D).flatten() 
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

    # dataset_name : 'existing_dataset' , 'real_time'


    images_1d, size_of_image, train_set_size, Path_to_images=find_train_images(dataset_name)

    mean1d=mean1d_(images_1d)
    images_mean=images_mean_(images_1d,mean1d, train_set_size)

    k = ((images_mean.shape[1])/2)+1

    eigenvectors,weights=PCA_(images_mean,k)

    return images_1d, mean1d, eigenvectors, weights, Path_to_images



### Test

def test(images_1d, mean1d, eigenvectors, weights, test_image, Threshold=10000):

    test_image = test_image.flatten()
    
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


### Opening videocamera for face recognition

def real_time_display(images_1d, mean1d, eigenvectors, weights,Path_to_images):
    face_cascade = cv2.CascadeClassifier('/home/pranjali/.local/lib/python2.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces= face_cascade.detectMultiScale(gray,1.2,5)

        for (x,y,w,h) in faces:
            face_img=img[y:y+h,x:x+w]
            face_image=face_img[:,:,0]

            face_image=cv2.resize(face_image,(40,40))

            closest_face_index, closest_face_distance = test(images_1d, mean1d, eigenvectors, weights, face_image)
            if closest_face_distance==-1:
                name_of_face_class=''
            else:
                list_of_images = copy.deepcopy(Path_to_images)
                name_of_face_class = os.path.basename(os.path.dirname(list_of_images[closest_face_index]))
                
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img, name_of_face_class, (x,y-2), cv2.FONT_HERSHEY_PLAIN, 2 ,(0,0,255),2)

        cv2.imshow("main",img)
        k = cv2.waitKey(30) & 0xff
        if k == 27 :
            break



    cap.release()
    cv2.destroyAllWindows()




#### Print help
def print_help():
    print(""" FACE RECOGNITION 

    Options : 

        1.  Train dataset for real-time and test it 
    

    Please enter an option (1) \n""")




def main():


    print_help()

    command = raw_input()

    if command == '1':
        
        ## train

        # creating real-time dataset
        dataset_name = 'real_time'
        print('Creating real-time dataset\n')
        
	print("If you want to click more images press y, if you want to use existing dataset press n")
	ans = raw_input()
	if(ans=='y'):
            print("please enter name of class")
            name =raw_input()
	    click_images(name)
            findface_haarcascades(name)
	
        # Training real-time dataset
        print('Training real-time dataset\n')
        images_1d, mean1d, eigenvectors, weights, Path_to_images = train(dataset_name)


        ## test real-time dataset
        print('Testing real-time dataset\n')
        print("starting video camera")
        real_time_display(images_1d, mean1d, eigenvectors, weights, Path_to_images)

        print("Done")


    else:
        print_help()
        


if __name__ == '__main__':
    main()






