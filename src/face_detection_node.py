#!/usr/bin/env python


import matplotlib.pyplot as plt
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Polygon, Point32

class FaceDetector:
    def __init__(self):
        self.bridge = CvBridge()
        # self.pub = rospy.Publisher('/bounding_boxes', Polygon, queue_size=1)
        self.pub_img = rospy.Publisher('/img_detect', Image, queue_size=1)
        self.sub = rospy.Subscriber('/webcam', Image, self.callback)
    
    def convertToRGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def detect_faces(self, f_cascade, colored_img, scaleFactor=1.1):
        img_copy = np.copy(colored_img)
        # convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

        # let's detect multiscale (some images may be closer to camera than others) images
        faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);

        # go over list of faces and draw them as rectangles on original colored img
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img_copy


    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            print(e)
            return

        haar_face_cascade = cv2.CascadeClassifier('/home/ziya/ds_ws/src/distributed_face_recognition/src/data/haarcascade_frontalface_alt.xml')
 

        # call our function to detect faces
        haar_detected_img = self.detect_faces(haar_face_cascade, cv_image)

        try:
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(haar_detected_img, "bgr8"))
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    rospy.init_node('facenet_object_detector', anonymous=True)
    facenet_object_detector = FaceDetector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")



	
