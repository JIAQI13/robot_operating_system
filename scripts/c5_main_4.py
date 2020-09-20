#!/usr/bin/env python
import math, time
from math import copysign

import rospy, cv2, cv_bridge, numpy
from tf.transformations import decompose_matrix, euler_from_quaternion
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
#from ros_numpy import numpify
import numpy as np

from kobuki_msgs.msg import Led
from kobuki_msgs.msg import Sound

import smach
import smach_ros

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseWithCovarianceStamped
#from ar_track_alvar_msgs.msg import AlvarMarkers
import tf
from sensor_msgs.msg import Joy

from nav_msgs.srv import SetMap
from nav_msgs.msg import OccupancyGrid

from detectshapes import ContourDetector
from detectshapes import Contour
from util import signal, rotate
import util

import work4

class Wait(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['start', 'end'])
        self.start = False

    def execute(self, userdata):
        joy_sub = rospy.Subscriber("joy", Joy, self.joy_callback)
        while not rospy.is_shutdown():
            if self.start:
                joy_sub.unregister()
                return 'start'
        joy_sub.unregister()
        return 'end'

    def joy_callback(self, msg):
        if msg.buttons[9] == 1: #start
            self.start = True
        print "start =", self.start

class Follow(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['running', 'work4'])

        self.bridge = cv_bridge.CvBridge()

        self.integral = 0
        self.previous_error = 0

        self.Kp = - 2 / 200.0
        self.Kd = 1 / 3000.0
        self.Ki = 0.0

        self.subscribered = False
        self.usb_cam_subscriber = None

        self.loop_start_time = None

    def execute(self, userdata):
        global g_twist_pub, g_full_red_line_count, current_twist, g_work4_returned

        if self.subscribered == False:
            self.subscribe()
            self.subscribered = True

        if self.loop_start_time == None:
            self.loop_start_time = time.time()

        if g_full_red_line_count == 2 and not g_work4_returned:
            twist = Twist()
            g_twist_pub.publish(twist)
            rospy.sleep(0.5)
            tmp_time = time.time()
            while time.time() - tmp_time < 1.85:
                g_twist_pub.publish(current_twist)
            g_twist_pub.publish(Twist())
            rotate(-35)
            tmp_time = time.time()
            while time.time() - tmp_time < 1.7:
                g_twist_pub.publish(current_twist)
            g_twist_pub.publish(Twist())
            
            self.usb_cam_subscriber.unregister()
            self.subscribered = False

            g_work4_returned = True
            return 'work4'

        else:
            if g_full_red_line_count == 3:
                g_work4_returned = False
            # start line
            if g_full_red_line_count == 4:
                g_full_red_line_count = 0
                g_twist_pub.publish(Twist())
                util.signal(2)
                self.loop_start_time = None

            g_twist_pub.publish(current_twist)
            return 'running'
    
    def subscribe(self):
        self.usb_cam_subscriber = rospy.Subscriber('usb_cam/image_raw', Image, self.usb_image_callback)
        print "Waiting for usb_cam/image_raw message..."
        rospy.wait_for_message("usb_cam/image_raw", Image)

    def usb_image_callback(self, msg):
        global g_full_red_line, g_half_red_line, g_full_red_line_count, current_twist

        count_full_red_line = False
        if g_full_red_line == False:
            count_full_red_line = True

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # white color mask
        lower_white = numpy.array([0, 0, 200])
        upper_white = numpy.array([360, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        h, w, _ = image.shape
        search_top = 3 * h / 4 + 20
        search_bot = 3 * h / 4 + 30
        white_mask[0:search_top, 0:w] = 0
        white_mask[search_bot:h, 0:w] = 0

        # red color mask
        lower_red = numpy.array([0, 100, 100])
        upper_red = numpy.array([360, 256, 256])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        h, w, _ = image.shape

        search_top = h - 40
        search_bot = h - 1

        red_mask[0:search_top, 0:w] = 0
        red_mask[search_bot:h, 0:w] = 0


        M = cv2.moments(white_mask)

        if M['m00'] > 0:
            self.cx_white = int(M['m10'] / M['m00'])
            self.cy_white = int(M['m01'] / M['m00'])
            cv2.circle(image, (self.cx_white, self.cy_white), 20, (0, 0, 255), -1)

            # BEGIN CONTROL
            err = self.cx_white - w / 2
            if g_full_red_line_count == 2 and not g_work4_returned:
                current_twist.linear.x = 0.5
                self.Kp = - 1 / 200.0
            else:
                self.Kp = - 2 / 200.0
                current_twist.linear.x = 0.7  # and <= 1.7

            self.integral = self.integral + err * 0.05
            self.derivative = (err - self.previous_error) / 0.05

            current_twist.angular.z = float(err) * self.Kp + (self.Ki * float(self.integral)) + (
                    self.Kd * float(self.derivative))

            self.previous_error = err
        else:
            self.cx_white = 0

        _, contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:

            for item in contours:
                area = cv2.contourArea(item)

                if area > 5000:
                    M = cv2.moments(item)
                    self.cx_red = int(M['m10'] / M['m00'])
                    self.cy_red = int(M['m01'] / M['m00'])
                    (x, y), radius = cv2.minEnclosingCircle(item)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(image, center, radius, (0, 255, 0), 2)
                    if self.cx_white == 0: #full red line
                        g_full_red_line = True
                        g_half_red_line = False
                        if count_full_red_line:
                            g_full_red_line_count +=1

                    elif x + radius < self.cx_white: #half red line
                        g_half_red_line = True
                        g_full_red_line = False
                    else:
                        g_full_red_line = False
                        g_half_red_line = False

            #cv2.imshow("refer_dot", image)
            #cv2.waitKey(3)

class SmCore:
    def __init__(self):

        self.sm = smach.StateMachine(outcomes=['end'])

        self.sis = smach_ros.IntrospectionServer('server_name', self.sm, '/SM_ROOT')
        self.sis.start()

        with self.sm:
            smach.StateMachine.add('Wait', Wait(),
                                    transitions={'end': 'end',
                                                'start': 'Follow'})
            smach.StateMachine.add('Follow', Follow(),
                                    transitions={'running':'Follow',
                                                'work4': 'SM_SUB_Work4'})

            # Create the sub SMACH state machine
            sm_sub_work4 = smach.StateMachine(outcomes=['end', 'returned'])
            # Open the container
            with sm_sub_work4:
                smach.StateMachine.add('PushBox', work4.PushBox(),
                                        transitions={'completed':'ON_RAMP',
                                                    'end':'end'})

                smach.StateMachine.add('ON_RAMP', work4.ON_RAMP(),
                                        transitions={'end':'end',
                                                    'returned':'returned'})

            smach.StateMachine.add("SM_SUB_Work4", sm_sub_work4,
                                    transitions={'end':'end',
                                                'returned':'Follow'})

    def execute(self):
        outcome = self.sm.execute()
        rospy.spin()
        self.sis.stop()

if __name__ == "__main__":

    g_full_red_line = False
    g_half_red_line = False
    current_twist = Twist()

    g_work4_returned = False

    g_full_red_line_count = 0

    rospy.init_node('c2_main')

    g_twist_pub = rospy.Publisher("/cmd_vel_mux/input/teleop", Twist, queue_size=1)

    c = SmCore()
    c.execute()
