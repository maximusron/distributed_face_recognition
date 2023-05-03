#!/usr/bin/env python3

import matplotlib.pyplot as plt
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Polygon, Point32
from pymongo import MongoClient
import pandas as pd
import csv


str_mdb = "mongodb+srv://ZiyaZainab:VGwI4vh6IBLUEetz@imagedetection.kqyfmwi.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(str_mdb)
dbname = client['Face_Recognition']
col1 = dbname['Bounding_Boxes']

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
        faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

        # go over list of faces and draw them as rectangles on original colored img
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return img_copy, faces


    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        haar_face_cascade = cv2.CascadeClassifier('/home/ziya/ds_ws/src/distributed_face_recognition/src/data/haarcascade_frontalface_alt.xml')
 

        # call our function to detect faces
        haar_detected_img, faces = self.detect_faces(haar_face_cascade, cv_image)
        # print(hello)

        now = rospy.get_rostime()
        # rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
        face_list = []
        for (x, y, w, h) in faces:
            item_1 = {
                "Timestamp": now.secs,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }
            face_list.append(item_1)

        print(len(face_list), "Faces detected")
        print("Uploading Face coordinates to Cloud...") 
        col1.insert_many(face_list)
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(haar_detected_img, "bgr8"))
    
if __name__ == '__main__':
    rospy.init_node('facenet_object_detector', anonymous=True)
    facenet_object_detector = FaceDetector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")








	
