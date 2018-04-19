# import packages
from imutils.video import FPS
import numpy as np
import argparse
import cv2
from dogClassify import getBreed, getBreedLabels
import tensorflow as tf


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

# Add arg for video file
ap.add_argument("--file", required=True, help = "path to video file")
args = ap.parse_args()

# READ VIDEO
#vid = cv2.VideoCapture('../../OpenCV/selfDrivingCar/solidWhiteRight.mp4')
vid = cv2.VideoCapture(args.file)
ret, im = vid.read()
imshape = im.shape

# -------CREATE VIDEO WRITER------------------
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#name = 'output/detectVideo_Output_' + args.method + '.avi'
videoOut = cv2.VideoWriter('detectPups_Output.avi',fourcc, 20.0, (im.shape[1],im.shape[0]))

# LOAD MODEL
print("[INFO] loading model...")
#print(args.method)
#print(type(args.method))
  
# Mobile net SSD Model from Caffe
prtxt = '../MobileNetSSD_deploy.prototxt.txt'
model = '../MobileNetSSD_deploy.caffemodel' 
    
# initialize the list of class labels MobileNet SSD was trained to
# detect with the caffe model.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
    
# Load the Model
net = cv2.dnn.readNetFromCaffe(prtxt, model)

# Randomize some colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES)+5, 3))
color = (165,255,0)

# Confidence threshold
conf = .7

# Labels file for different breeds
label_lines = getBreedLabels('./dog_labels.txt')
breedModel = './dog_graph.pb'
 
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video play...")
fps = FPS().start()

# Run image of detected dog through retrained inception graph to get breed
with tf.gfile.FastGFile(breedModel, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

# Begin tf session
with tf.Session() as sess:

    # tensor to output dog breed
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    # frame counter
    frameCount = 1
    
    # Loop through video       
    #for frameNum in range(1,len(test_images)-1):
    while True:
        
        # Watch it
        print("Frame: ", frameCount)
        frameCount += 1
    
        # read next im of video
        #frame = mpimg.imread('../selfDrivingCar/myStuff/test_video/videoFrames/'+str(frameNum)+'.jpg')
        ret,frame = vid.read()
       # b,g,r = cv2.split(frame) # get rgb channels from matplotlib image
        #frame = cv2.merge((r,g,b)) # merge three channels to create openCV color image
        
        # Proceed if we got an image
        if ret:
        	# grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (350, 350)),
        		0.007843, (350, 350), (127.5, 127.5, 127.5), swapRB=True, crop=False)
         
         	# pass the blob through the network and obtain the detections and
        	    # predictions
            net.setInput(blob)
            detections = net.forward()
        
        	    # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                
        		# extract the confidence (i.e., probability) associated with
        		# the prediction
                confidence = detections[0, 0, i, 2]
         
        		# filter out weak detections by ensuring confidence
        		# greater than the minimum confidence
                if confidence > conf: #args["confidence"]:
        			# extract the index of the class label from the
        			# `detections`, then compute the (x, y)-coordinates of
        			# the bounding box for the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    if CLASSES[idx] == 'dog':
                       # print("dog")
                            
                        # Crop image of dog
                        if startX < 0: startX = 0
                        if startY < 0: startY = 0
                        if endX < 0: endX = 0
                        if endY < 0: endY = 0
                        dogCrop = frame[startY:endY, startX:endX]
                        
                        
                        # Run the image through the softmax to get predictions
                        predictions = sess.run(softmax_tensor, \
                            {'DecodeJpeg:0': dogCrop})
    
                        # Sort to show labels of first prediction in order of confidence
                        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]   
    
                        # Get predicted breed
                        predicted_breed = label_lines[top_k[0]]
                        
                        # Get dog breed by running cropped image through classifier
                        #dogBreed = getBreed(dogCrop, breedLabels, breedModel)
                        
                        # get label to put on image
                        label = "{}: {:.2f}%".format(predicted_breed,
        				          confidence * 100)
                        
                        # Draw rectangle around dog
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
        			     	color, 2); 
                        
                        # Put label above boxed dog
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
    				     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        	# show the output frame
           # cv2.imshow("Frame", frame)
            #key = cv2.waitKey(1) & 0xFF
            
            # write the frame to output vid
            videoOut.write(frame)
         
        	# if the `q` key was pressed, break from the loop
            #if key == ord("q"):
            #    break
         
        	# update the FPS counter
            fps.update()
        
        # if no image/at end of video break
        else:
            break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()

# Release video
videoOut.release()
