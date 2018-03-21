#use Deep Neural Networks to determine the quality of your face training data set
#useful for when you want to do projects with face training of individuals and
#you want to see how good your data sets are

# ex: python eval_face_training.py -d training_images -p deploy.prototxt.txt
#        -m facedetection.caffemodel

#Sample output:
#Evaluating directory: training_images
#   Total dirs: 2
#   Total files: 24
#   Total faces: 22
#   Total images: 24
#   Total good images: 22
#   Overall score: 91.6666666667%
#   Execution time: 1.46788811684 seconds (0.0611620048682 sec avg per file)

import cv2          #use OpenCV for face detection

import os           #for file/dir handling
import numpy as np  #for handling arrays
import argparse     #for parsing arguments passed to program
import sys          #for return codes

import time         #for execution timing purposes

#function to detect face using OpenCV and Deep Neural Networks (DNN)
def detect_faces(img, net, args):
    faces = [] #will hold whatever faces we detect
    (h, w) = img.shape[:2]
    # Convolutional Neural Network expects an image that is 300x300, so scale
    # the image to fit that and set scalefactor to 1.0
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network (going thru mean subtraction, normalizing,
    # and channel swapping) to obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections, filter out anything with a confidence not hi enough
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the prediction
    	confidence = detections[0, 0, i, 2]

    	# filter out weak detections by ensuring the `confidence` is
    	# greater than the minimum confidence
        if confidence >= args["confidence"]:
    		# compute the (x, y)-coordinates of the bounding box for the
    		# object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_width = (endX - startX)
            face_height = (endY - startY)
            faces.append((startX, startY, face_width, face_height, confidence))

    #if multifaces is set in arguments and there are multiple, treat the one
    #with highest confidence and return it as the single face in the image
    if args["multifaces"] and len(faces) > 1:
        faces = sorted(faces, key=lambda a_entry: a_entry[4], reverse=True)
        faces = [faces[0]]

    return faces

#function to draw rectangle on an image
#given the x/y coordinates and given width/height in the rect
def draw_red_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def eval_training_data(net, args):
    total_dirs = 0
    total_files = 0
    total_faces = 0
    total_images = 0
    total_good_images = 0  #an image with only 1 face

    # go through all subdirectories / files and attempt to look at images...
    for root, dirs, files in os.walk(args["dir"]):
        total_dirs += len(dirs)
        total_files += len(files)
        for file in files:
            if args["verbose"]:
                print("loading file: " + root + "/" + file)

            image = cv2.imread(root + '/' + file) #attempt to load image...

            if image is None: #skip unrecognized images or file`s...
                if args["verbose"]:
                    print("   not a recognized image...")
                continue

            total_images += 1

            faces = detect_faces(image, net, args)
            total_faces += len(faces)

            if args["verbose"]:
                print("   {0} faces".format(len(faces)))

            #display an image window to show the image
            if len(faces) == 1:
                total_good_images += 1
            elif args["display_problems"]:  #faces != 1 (not good)
                for face in faces:
                    print(face)
                    draw_red_rectangle(image, face[:4])
                cv2.imshow("Bad Training Data: " + file, cv2.resize(image, (400, 500)))
                cv2.waitKey(0)

    #pass back all the totals info for displaying stats
    return total_dirs, total_files, total_faces, total_images, total_good_images


#parse the arguments passed into the program
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True,
	help="path to training images")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to the Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to the Caffe pre-trained model (ending .caffemodel)")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum confidence percent to filter out weak face detections, 1.0 max")
ap.add_argument("-f", "--multifaces", action="store_true",
	help="if set and there is multiple faces, will treat highest one as good")
ap.add_argument("--display_problems", action="store_true",
	help="will show a window of each failed image that has 0 or > 1 faces in it")
ap.add_argument("-v", "--verbose", action="store_true",
	help="turn on verbose output (each file parsed, etc)")
args = vars(ap.parse_args())

print("Evaluating directory: " + args['dir'])

#load the prototxt and model, throws an error if there is a problem
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

#now actually evaluate the training data and time how long it takes
start_time = time.time()
dirs, files, faces, images, good_images = eval_training_data(net, args)
total_time = time.time() - start_time

#print totals and stats
print("   Total dirs: %s" % dirs)
print("   Total files: %s" % files)
print("   Total faces: %s" % faces)
print("   Total images: %s" % images)
print("   Total good images: %s" % good_images)
if (files > 0):
    score = float(good_images * 100) / float(images)
    print('   Overall score: {0}%'.format(score))
    print('   Execution time: {0} seconds ({1} sec avg per file)'.format(total_time,
        (total_time / files)))

cv2.destroyAllWindows() #cleanup