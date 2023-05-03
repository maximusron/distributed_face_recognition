#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
import argparse

pub = rospy.Publisher('/webcam', Image,queue_size=10)


def main():
    cap = cv2.VideoCapture(0)
    k = 0
    while(1):
        _, frame = cap.read()
        rospy.loginfo("Publishing") 
        print("Publishing{}".format(k))
        br = CvBridge()
        pub.publish(br.cv2_to_imgmsg(frame))
        rate.sleep()
        k = k+1
        

if __name__ == '__main__':  
    rospy.init_node("webcam_node", anonymous=True)
    rate = rospy.Rate(30)
    main()
    rospy.spin()
