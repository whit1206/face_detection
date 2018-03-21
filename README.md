# Face Detection Tools

These tools are useful for working with face detection using OpenCV. They are writtin in Python 3 but should work with Python 2 relatively easily. I plan to continue to work and improve this collection, but would appreciate any feedback.


### Prerequisites

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
  
## Authors

* **Chris Whitten** - *Initial work* - [Github](https://github.com/whit1206)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
Inspiration:
* [Siraj Raval](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) - For dispensing incredible knowledge on computer vision and machine learning
* [Adrian at PyImageSearch](http://www.pyimagesearch.com) - For great learning material on how to use OpenCV and use DNNs
