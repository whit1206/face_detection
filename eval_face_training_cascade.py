#use Haar Cascades to determine the quality of your face training data set
#useful for when you want to do projects with face training of individuals and
#you want to see how good your data sets are
# Note the overall score you get will greatly change with the type of Haar
#      casecade you select. haarcascade_frontalface_default.xml that comes
#      with Python seemed to have the best results with testing

# ex: python eval_face_training_cascade.py -d training_images
#            -p haarcascade_frontalface_default.xml

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
def detect_faces(img, cascade_classifier, args):
    faces = [] #will hold whatever faces we detect

    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #get all face rectangles that pass through the casecade
    #note: for now the way detectMultiScale works with rejectLevels seems
    #      to be not working fully. It only returns objects that go through
    #      the whole cascade. The function detectMultiScale3 does return
    #      rejectLevels but it seems to not be implemented fully for now
    #      see: https://github.com/opencv/opencv/issues/8016
    faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.2,
                minNeighbors=5, minSize=(40,40), flags = cv2.CASCADE_SCALE_IMAGE);

    return faces

#function to draw rectangle on an image
#given the x/y coordinates and given width/height in the rect
def draw_red_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def eval_training_data(cascade_classifier, args):
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

            faces = detect_faces(image, cascade_classifier, args)
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
ap.add_argument("-c", "--cascade", required=True,
	help="path to the Cascade Classifier file")
ap.add_argument("--display_problems", action="store_true",
	help="will show a window of each failed image that has 0 or > 1 faces in it")
ap.add_argument("-v", "--verbose", action="store_true",
	help="turn on verbose output (each file parsed, etc)")
args = vars(ap.parse_args())

print("Evaluating directory: " + args['dir'])

#now actually evaluate the training data and time how long it takes
start_time = time.time()
cascade_classifier = cv2.CascadeClassifier(args["cascade"]) #load OpenCV face detector

if cascade_classifier.empty():
    cv2.destroyAllWindows()
    sys.exit("Failed to load cascade_classifier: " + args["cascade"])

dirs, files, faces, images, good_images = eval_training_data(cascade_classifier,
                                                             args)
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