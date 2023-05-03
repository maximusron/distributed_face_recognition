# Import Libraries
import cv2
import numpy as np

#!/usr/bin/env python3

import matplotlib.pyplot as plt
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pymongo import MongoClient



GENDER_MODEL = 'weights/deploy_gender.prototxt'
GENDER_PROTO = 'weights/gender_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"


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
        self.pub_img = rospy.Publisher('/img_detect', Image, queue_size=1)
        self.sub = rospy.Subscriber('/webcam', Image, self.callback)
        self.str = "mongodb+srv://Ziya:Dsproj23@sample.dq06bsy.mongodb.net/?retryWrites=true&w=majority"
        self.client = MongoClient(str)
        self.dbname = client['Gender Estimation']
        self.col1 = dbname['Boxes + Gender']



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
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
            # image --> Input image to preprocess before passing it through our dnn for classification.
            # scale factor = After performing mean substraction we can optionally scale the image by some factor. (if 1 -> no scaling)
            # size = The spatial size that the CNN expects. Options are = (224*224, 227*227 or 299*299)
            # mean = mean substraction values to be substracted from every channel of the image.
            # swapRB=OpenCV assumes images in BGR whereas the mean is supplied in RGB. To resolve this we set swapRB to True.
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

            # Display processed image
        display_img("Gender Estimator", frame)
        # uncomment if you want to save the image
        cv2.imwrite("output.jpg", frame)
        # Cleanup
        cv2.destroyAllWindows()

        try:
            now = rospy.get_rostime()
            rospy.loginfo("Current time %i %i", now.secs, now.nsecs)

            for (x, y, w, h) in faces:
                item_1 = {
                    "Timestamp": now.secs,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
                self.col1.insert_one([item_1])

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








	
