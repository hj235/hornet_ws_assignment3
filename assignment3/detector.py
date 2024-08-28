import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage

import numpy as np
import cv2
from cv_bridge import CvBridge

class Detector(Node):

    def __init__(self):
        super().__init__("detector")
        self.pub_debug_img = self.create_publisher(Image, "/detected/debug_img", 10)
        self.sub_image_feed = self.create_subscription(
            CompressedImage,
            "/left/compressed",
            self.image_feed_callback,
            10)
        self.bridge = CvBridge()

    def image_feed_callback(self, msg):
        # Feel free to modify this callback function, or add other functions in any way you deem fit

        # Here is sample code for converting a coloured image to gray scale using opencv
        cv_img = self.bridge.compressed_imgmsg_to_cv2(msg)
        img_height = cv_img.shape[:1][0]
        # transformed_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        # img_msg = self.bridge.cv2_to_imgmsg(transformed_img, encoding="mono8")

        # self.pub_debug_img.publish(img_msg)

        img_tfm = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        img_tfm = cv2.GaussianBlur(img_tfm, (5, 5), 0)

        # get Red mask
        upperRed = cv2.inRange(img_tfm, (0, 100, 10), (13, 255, 255))
        lowerRed = cv2.inRange(img_tfm, (170, 100, 10), (180, 255, 255))
        redMask = cv2.bitwise_or(upperRed, lowerRed)

        # get Orange mask
        # orangeMask = cv2.inRange(img_tfm, (6, 100, 20), (13, 255, 255))
        
        kernel = np.ones((4,4),np.uint8)
        red_open = cv2.morphologyEx(redMask, cv2.MORPH_OPEN, kernel)
        red_close = cv2.morphologyEx(red_open, cv2.MORPH_CLOSE, kernel)
        # orange_open = cv2.morphologyEx(orangeMask, cv2.MORPH_OPEN, kernel)
        # orange_close = cv2.morphologyEx(orange_open, cv2.MORPH_CLOSE, kernel)

        red_ctrs, hierarchy = cv2.findContours(red_close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # fill in the holes in the contours (to eliminate rectangles within rectangles)
        for c in red_ctrs:
            cv2.drawContours(red_close, [c], 0, 255, -1)
        red_ctrs, hierarchy = cv2.findContours(red_close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # find small contours and dilate them (to try to join up small and nearby contours)
        tiny_ctrs = [c for c in red_ctrs if cv2.contourArea(c) < 100]
        dark_bg = np.zeros(red_close.shape, np.uint8)
        cv2.drawContours(dark_bg, tiny_ctrs, -1, 255, 3)
        dilated_spots = cv2.dilate(dark_bg, np.ones((6,6),np.uint8))

        red_close = cv2.bitwise_or(dilated_spots, red_close)
        red_ctrs, hierarchy = cv2.findContours(red_close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in red_ctrs:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            # disregard rectangles whose base is below half the image
            if y+h < img_height/2: 
                continue 
            cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0,0,255), 3)
            cv2.putText(cv_img, "red", (x+10, y-10), 0, 0.7, (0,0,255))

        # orange_ctrs, hierarchy = cv2.findContours(orange_close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # for c in orange_ctrs:
        #     rect = cv2.boundingRect(c)
        #     x, y, w, h = rect
        #     cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0,0,255), 3)
        #     cv2.putText(cv_img, "orange", (x+10, y-10), 0, 0.7, (0,165,255))
        # # cv2.drawContours(cv_img, contours, -1, (0, 255, 0), 3)

        img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")

        self.pub_debug_img.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    detector = Detector()
    rclpy.spin(detector)

    # Below lines are not strictly necessary
    detector.destroy_node()
    rclpy.shutdown()
        
if __name__=='__main__':
    main()
