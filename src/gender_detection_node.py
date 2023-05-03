#!/usr/bin/env python3

import matplotlib.pyplot as plt
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pymongo import MongoClient
# Import Libraries
import cv2
import numpy as np

passwd = "VGwI4vh6IBLUEetz" 
str_mdb = "mongodb+srv://ZiyaZainab:" + passwd + "@imagedetection.kqyfmwi.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(str_mdb)
dbname = client['Gender_Detection']
col1 = dbname['Gender']

GENDER_MODEL = '/home/ziya/ds_ws/src/distributed_face_recognition/src/weights/deploy_gender.prototxt'
GENDER_PROTO = '/home/ziya/ds_ws/src/distributed_face_recognition/src/weights/gender_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
FACE_PROTO = "/home/ziya/ds_ws/src/distributed_face_recognition/src/weights/deploy.prototxt.txt"
FACE_MODEL = "/home/ziya/ds_ws/src/distributed_face_recognition/src/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"


face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def display_img(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_faces(frame, confidence_threshold=0.5):

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    face_net.setInput(blob)
    output = np.squeeze(face_net.forward())
    faces = []

    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def get_optimal_font_scale(text, width):
    """Determine the optimal font scale based on the hosting frame width"""
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)


class FaceDetector:
    def __init__(self):
        self.bridge = CvBridge()
        # self.pub = rospy.Publisher('/bounding_boxes', Polygon, queue_size=1)
        self.pub_img = rospy.Publisher('/gender_detect', Image, queue_size=1)
        self.sub = rospy.Subscriber('/webcam', Image, self.callback)



    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            print(e)
            return
        
        frame_width = 1080
        img = cv_image
        # resize the image, uncomment if you want to resize the image
        frame = img.copy()
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        # predict the faces
        faces = get_faces(frame)
        # Loop over the faces detected
        # for idx, face in enumerate(faces):
        gender_list = []
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
            blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
                227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
            # Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            i = gender_preds[0].argmax()
            gender = GENDER_LIST[i]
            gender_confidence_score = gender_preds[0][i]
            # Draw the box
            label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
            gender_list.append((start_x, start_y, gender, gender_confidence_score*100))
            print(label)
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
            # get the font scale for this image size
            optimal_font_scale = get_optimal_font_scale(label,((end_x-start_x)+25))
            box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
            # Label processed image
            cv2.putText(frame, label, (start_x, yPos),
                        cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, box_color, 2)

        try:
            now = rospy.get_rostime()
            rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
            db_gender_list = [] 
            for (x, y, gender, prob) in gender_list:
                item_1 = {
                    "Timestamp": now.secs,
                    "x": int(x),
                    "y": int(y),
                    "Gender": gender,
                    "Probability": prob
                }
                db_gender_list.append(item_1)
            print(len(db_gender_list), "Faces detected")
            print("Uploading Face coordinates to Cloud...") 
            col1.insert_many(db_gender_list)
            self.pub_img.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    rospy.init_node('gender_detect_node', anonymous=True)
    facenet_object_detector = FaceDetector()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")








	
