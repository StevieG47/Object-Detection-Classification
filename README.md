## Object Detection Using Pre-trained models
Using OpenCV's dnn module with pre-trained models from tensorflow and caffe
```
python detectVideo.py --method caffe --file office.mp4
```
![office](https://user-images.githubusercontent.com/25371934/35825480-f53269b0-0a83-11e8-8974-f3cde1eee59f.gif)

This will show the object detection on screen and output a video in the output directory. Using detectImage on an image file will output an image in the output directory.

## Custom Object Detection
Using a custom trained object detector, a stop sign detector and bison detector were done following the tutorial [here](https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/   ). The detectImage and detectVideo scripts can use these detectors instead of the pre-trained caffe/tensorflow ones.
```
python detectImage.py --method trafficSign --image images/stopSign.jpg
```
![detectimage-output](https://user-images.githubusercontent.com/25371934/37247584-5267cf8c-248b-11e8-8c63-1fbe55502e41.png)

```
python detectVideo.py --method trafficSign --file stopSign.mp4
```
![stopgif](https://user-images.githubusercontent.com/25371934/37247823-9ac8dd08-248f-11e8-8585-0362382b3bfb.gif)


```
python detectImage.py --method bison --image images/bison.jpg
```
![detectimage-output1](https://user-images.githubusercontent.com/25371934/35824584-4e30daa4-0a81-11e8-980e-d5a8ace657a4.png)


```
python detectVideo.py --method bison --file bison.mp4
```

![bison](https://user-images.githubusercontent.com/25371934/35825977-67501596-0a85-11e8-96f0-d6f37b055fa0.gif)

## Notes
- Arguments for detectVideo are --method, which is caffe, tensorflow, trafficSign or bison, and --file, which is the path to the video
- Arguments for detectImage are --method, which is caffe, tensorflow, trafficSign or bison, and --image, which is the path to the image
- Arguments for detectWebcam are --method, which is caffe or tensorflow
- The training for the bison detector had to be stopped early due to an error, so as of right now it is not great (it will think pretty much anything is a bison)
- The stop sign model does better than the bison one, but is still far from perfect

## References
https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/
https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/     
https://github.com/tensorflow/models
