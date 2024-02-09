import cv2
import numpy as np
import random
import json
from simple_pid import PID

action_dict = {1: 'go_ahead',2: 'turn_left', 3: 'turn_right', 4: 'stop'}

signal = 'straight'
signal_dict = {}

current_action = action_dict[1]
next_action = action_dict[1]

def _birdview_transform(img):
    """Apply bird-view transform to the image
    """
    IMAGE_H = 480
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    return warped_img

def _find_left_right_points(image, draw=None, percent_im_height= 0.9):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_height, im_width = image.shape[:2]

    # Consider the position 70% from the top of the image
    interested_line_y = int(im_height * percent_im_height)
    if draw is not None:
        cv2.line(draw, (0, interested_line_y),
                 (im_width, interested_line_y), (0, 0, 255), 2)
    interested_line = image[interested_line_y, :]

    # Detect left/right points
    left_point = -1
    right_point = -1
    lane_width = 100
    center = im_width // 2

    # Traverse the two sides, find the first non-zero value pixels, and
    # consider them as the position of the left and right lines
    for x in range(center, 0, -1):
        if interested_line[x] != 255:
            left_point = x
            break
    for x in range(center + 1, im_width):
        if interested_line[x] != 255:
            right_point = x
            break

    # Predict right point when only see the left point
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width

    # # Predict left point when only see the right point
    # if right_point != -1 and left_point == -1:
    #     left_point = right_point - lane_width

    # Draw two points on the image
    
    if draw is not None:
        if left_point != -1:
            draw = cv2.circle(
                draw, (left_point, interested_line_y), 7, (1), -1)
        if right_point != -1:
            draw = cv2.circle(
                draw, (right_point, interested_line_y), 7, (1), -1)

    return left_point, right_point

def _find_lane_center(img):
    gray = img
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5) 
    dist_output = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX) 
    return dist_output

def _convert_segment_to_img(lane_segment):
    # img = cv2.cvtColor(lane_segment*255, cv2.COLOR_GRAY2BGR)
    img = lane_segment*255
    img = img.astype(dtype='uint8')
    return img

def _drive_in_straight_lane(img_birdview):
    distance_Transform_img = _find_lane_center(img_birdview)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(distance_Transform_img[345:350,:])
    im_center = img_birdview.shape[1] // 2

    distance_Transform_img = cv2.circle(
        distance_Transform_img, (max_loc[0], 345), 7, (0), -1)
    distance_Transform_img = cv2.circle(
        distance_Transform_img, (max_loc[0], 350), 7, (0), -1)

    pid = PID(0.004, 0, 0, setpoint=0)
    pid.output_limits = (-1, 1)

    center_diff = im_center - max_loc[0]
    steering_angle = pid(center_diff)
    # cv2.imshow('_find_lane_center',distance_Transform_img)
    return steering_angle, distance_Transform_img

def _signal_trigger(signal_obj):
    # print(signal_obj)
    global signal
    global signal_dict


    # if (signal_obj[0][1] > 0.70) and (signal != signal_obj[0][0]):
        # signal = signal_obj[0][0]
        # print(signal, signal_obj[0][1])
    
    if (signal != signal_obj[0][0]):
        signal_dict[signal_obj[0][0]] = signal_dict.get(signal_obj[0][0], 0) + 1

        # print(signal_dict)
        for i in signal_dict.keys():
            if signal_dict[i] >= 10:
                signal = signal_obj[0][0]
                print(signal + ' sign detected!')
                signal_dict.clear()
                break

throttle = 0.12
steering_angle = 0
def car_control(signal_obj, lane_segment, isShow= False):
    global signal
    global current_action
    global throttle
    global steering_angle
    if signal_obj != None:
        _signal_trigger(signal_obj)

    if type(lane_segment) != type(None):
    # if len(lane_segment) != 0:

        img = _convert_segment_to_img(lane_segment[1])
        im_height, im_width = img.shape[:2]
        draw = img.copy()
        img_birdview = _birdview_transform(img)
        draw[:, :] = _birdview_transform(draw)

        points_x_left = int(im_width/10*3.3)
        points_x_right = int(im_width/10*6.7)
        points_y_1 = int(im_height * 0.75)
        points_y_2 = int(im_height * 0.85)

        point_delta_detect_obj = 50
        points_x_left_detect_obj = int(points_x_left + point_delta_detect_obj)
        points_x_right_detect_obj = int(points_x_right - point_delta_detect_obj)

        distance_Transform_img = None

        draw = cv2.circle(
            draw, (points_x_left, points_y_1), 7, (255), -1)
        draw = cv2.circle(
            draw, (points_x_right, points_y_1), 7, (255), -1)

        draw = cv2.circle(
            draw, (int(im_width *0.5), int(im_height * 0.5)), 7, (0), -1)

        draw = cv2.circle(
            draw, (points_x_left_detect_obj, points_y_1), 7, (0), -1)
        draw = cv2.circle(
            draw, (points_x_right_detect_obj, points_y_1), 7, (0), -1)
        
        draw = cv2.circle(
            draw, (points_x_left_detect_obj, points_y_2), 7, (0), -1)
        draw = cv2.circle(
            draw, (points_x_right_detect_obj, points_y_2), 7, (0), -1)
        
        

        if signal == 'left' and img_birdview[points_y_1][points_x_left] == 255:
            current_action = action_dict[2]
        elif signal == 'right' and img_birdview[points_y_1][points_x_right] == 255:
            current_action = action_dict[3]
        elif signal == 'stop':
            current_action = action_dict[4]
        turning_steering_angle = 0.65
        if current_action == action_dict[1]:
            steering_angle, distance_Transform_img = _drive_in_straight_lane(img_birdview)
            if abs(steering_angle) <= 0.08:
                # is_turn_left = False
                is_turn_right = False

                if img_birdview[points_y_1][points_x_left_detect_obj] != 255:
                    # print('turn right')
                    steering_angle += 0.3
                    is_turn_right = True

                if img_birdview[points_y_1][points_x_right_detect_obj] != 255 and is_turn_right == False:
                    # print('turn left')
                    steering_angle -= 0.3

        elif current_action == action_dict[2]:
            steering_angle = -turning_steering_angle
            if img_birdview[points_y_1][points_x_left] != 255 and img_birdview[points_y_1][points_x_right] != 255:
                signal = 'straight'
                current_action = action_dict[1]

        elif current_action == action_dict[3]:
            steering_angle = turning_steering_angle
            if img_birdview[points_y_1][points_x_right] != 255:
                signal = 'straight'
                current_action = action_dict[1]

        elif current_action == action_dict[4]:
            steering_angle, distance_Transform_img = _drive_in_straight_lane(img_birdview)
            stop_line_detect_img = cv2.cvtColor(lane_segment[0], cv2.COLOR_BGR2GRAY)
            if stop_line_detect_img[im_height-150][int(im_width/3*1)] > 200 and stop_line_detect_img[im_height-150][int(im_width/3*2)] > 200:
                print('STOP!')
                throttle = 0
            # if throttle < 0: throttle = 0
            # print(throttle)
            

        if isShow:
            cv2.imshow('draw', draw)
            if distance_Transform_img != None:
                cv2.imshow('find lane center', distance_Transform_img)
            cv2.waitKey(1)

        # steering_angle = max(-1, min(1, steering_angle))
        return json.dumps({"throttle": throttle, "steering": steering_angle})
        # return None