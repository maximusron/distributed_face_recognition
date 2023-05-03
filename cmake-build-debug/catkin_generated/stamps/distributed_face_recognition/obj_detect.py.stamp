#!/usr/bin/env python3.8

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
import torch
from torchvision import transforms
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Polygon, Point32
from facenet_pytorch import MTCNN

# Load the FaceNet MTCNN model
mtcnn = MTCNN()

class FaceNetObjectDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/bounding_boxes', Polygon, queue_size=1)
        self.pub_img = rospy.Publisher('/img_detect', Image, queue_size=1)
        self.sub = rospy.Subscriber('/webcam', Image, self.callback)

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # Pre-process the image for FaceNet
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Run the FaceNet MTCNN model
        with torch.no_grad():
            boxes, _ = mtcnn.detect(img_rgb)

        # Draw bounding boxes on the image and publish
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.astype(np.int32)

                bounding_box = Polygon()
                bounding_box.points.append(Point32(x1, y1, 0))
                bounding_box.points.append(Point32(x2, y1, 0))
                bounding_box.points.append(Point32(x2, y2, 0))
                bounding_box.points.append(Point32(x1, y2, 0))
                self.pub.publish(bounding_box)

                # Draw bounding box on the image
                cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Publish the result image with bounding boxes
        try:
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    rospy.init_node('facenet_object_detector', anonymous=True)
    facenet_object_detector = FaceNetObjectDetector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
