# Face Detection Tools

These tools are useful for working with face detection using OpenCV. They are writtin in Python 3 but should work with Python 2 relatively easily. I plan to continue to work and improve this collection, but would appreciate any feedback.


## Prerequisites

Assumes you have the following installed:
* Python 3
* OpenCV
* NumPy

## Tools, What They Do, and How to Run Each of Them

These scripts provide different functionality or use different algorithms:
* **eval_face_training.py** - Uses Deep Neural Networks to evaluate your face training dataset. It counts a good face image an image with a single face with a confidence factor that you set (defaults to 50%). The an example and parameters are as follows: python eval_face_training.py -d training_images -p deploy.prototxt.txt -m facedetection.caffemodel
  * --dir (-d) [path]     : The path to the training data to look at
  * --prototxt (-p) [file]: Path to the Caffe 'deploy' prototxt file
  * --model (-m) [file]   : Path to the Caffe pre-trained model (ending .caffemodel)
  * --confidence (-c) int : Minimum confidence percent to filter out weak face detections, 1.0 max
  * --multifaces (-f)     : If set and there is multiple faces, will treat highest one as the good one
  * --display_problems    : Will show a window of each failed image that has 0 or >1 faces in it (faces in a red rectangle)
  * --verbose (-f)        : Turn on verbose output (each file parsed, etc)
* **eval_face_training_cascade.py** - Uses Haar Cascades to evaluate face training data (like eval_face_training.py). The example and parameters are as follows: python eval_face_training_cascade.py -d training_images -p haarcascade_frontalface_default.xml
  * --dir (-d) [path]     : The path to the training data to look at
  * --cascade (-c) [file] : Path to the face Haar Cascade file to use
  * --display_problems    : Will show a window of each failed image that has 0 or >1 faces in it (faces in a red rectangle)
  * --verbose (-f)        : Turn on verbose output (each file parsed, etc)
  
## Sample Output, Using These Tools

I like to use the DNN based face data evaluators for finding good face training data of individuals. For me to set that up:
1. Download images that I want to train with into a local directory
2. Run the python eval_face_trainin.py file against that directory and it will tell me overall how many images it thinks are "good" (one face meeting the confidence factor): 
`python eval_face_training.py -d training_images -p deploy.prototxt.txt -m res10_300x300_ssd_iter_140000.caffemodel`
3. The output will be something like this:
```
Evaluating directory: training_images
   Total dirs: 2
   Total files: 24
   Total faces: 23
   Total images: 24
   Total good images: 21
   Overall score: 87.5%
   Execution time: 1.54642605782 seconds (0.0644344190756 sec avg per file)
```
4. I can see overall most of the images were good (yeah!) but I may want to throw some out. If I specify verbose (-v) then it will tell me specifically for each file how it did:
```
   ...
loading file: training_images/s2/9.jpg
   1 faces
loading file: training_images/s2/12.jpg
   0 faces
loading file: training_images/s2/11.jpg
   1 faces
loading file: training_images/s2/10.jpg
   ...
 ```
I can now remove/replace those files if I wish, or I can turn on '--display_problems' flag and run again and it will show me the images. This is also useful if I get an image that has more then 1 face in it.

## Authors

* **Chris Whitten** - *Initial work* - [Github](https://github.com/whit1206)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
Inspiration:
* [Siraj Raval](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) - For dispensing incredible knowledge on computer vision and machine learning
* [Adrian at PyImageSearch](http://www.pyimagesearch.com) - For great learning material on how to use OpenCV and use DNNs
