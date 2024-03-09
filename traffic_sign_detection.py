import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def get_boxes_from_mask(mask):
    """
    Find the boxes of traffic signs from mask
    """
    bboxes = []

    nccomps = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    numLabels, labels, stats, centroids = nccomps
    
    for i in range(numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        bboxes.append([x, y, w, h])
    return bboxes
    
def filter_bboxes(mask, bboxes):
    im_height, im_width = mask.shape[:2]
    return_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        # filter bbox if it's too small
        if w < 20 or h < 20:
            continue
        
        # filter bbox if it's too big
        if w > 0.2 * im_width or h > 0.2 * im_height:
            continue
        # filter bbox if its height and width are too different
        if w / h > 1.3 or h / w > 1.3:
            continue
        return_bboxes.append([x, y, w, h])
    return return_bboxes

def filter_signs_by_color(image):
    """
    filter the traffic signs according to HSV color
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # filter the red signs for stop signs
    lower1, upper1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower2, upper2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    mask_1 = cv2.inRange(image, lower1, upper1) # dải màu đỏ thứ nhất
    mask_2 = cv2.inRange(image, lower2, upper2) # dải màu đỏ thứ hai
    mask_r = cv2.bitwise_or(mask_1, mask_2) # kết hợp 2 kết quả từ 2 dải màu khác nhau

    # filter the red signs for navigation signs
    lower3, upper3 = np.array([100, 150, 0]), np.array([140, 255, 255])
    # lower3, upper3 = np.array([85, 50, 200]), np.array([135, 250, 250])
    lower4, upper4 = np.array([90, 60, 90]), np.array([220, 250, 150])
    mask_3 = cv2.inRange(image, lower3, upper3)
    mask_4 = cv2.inRange(image, lower4, upper4)
    mask_b = cv2.bitwise_or(mask_3, mask_4)
    # mask_b = mask_3

    # combine the results
    mask_final  = cv2.bitwise_or(mask_r,mask_b)
    return mask_final

def merge_intersection_box(bboxes):
    '''
    This function will merge 2 boxes if it is intersection.
    '''
    new_bboxes = []

    for i, bbox_i in enumerate(bboxes):
        is_change = False
        for j, new_bbox in enumerate(new_bboxes):
            x_i, y_i, w_i, h_i = bbox_i
            x_j, y_j, w_j, h_j = new_bbox
            intersect_x = list(set(range(x_i, x_i+w_i)) & set(range(x_j, x_j+w_j)))
            intersect_y = list(set(range(y_i, y_i+h_i)) & set(range(y_j, y_j+h_j)))
            len_intersect_x = len(intersect_x)
            len_intersect_y = len(intersect_y)

            if len_intersect_x > 0 and len_intersect_y > 0:
                x = min(x_i, x_j)
                y = min(y_i, y_j)
                w = w_i + w_j - (max(intersect_x) - min(intersect_x))
                h = h_i + h_j - (max(intersect_y) - min(intersect_y))

                new_bboxes[j] = [x, y, w, h]
                is_change = True
                break
        
        if not is_change: new_bboxes.append(bbox_i)

    return new_bboxes


def detect_traffic_signs(img, model, draw=None):
    # hsvs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsvs', hsvs)
    filtered_img = filter_signs_by_color(img)
    cv2.imshow('filtered_img', filtered_img)
    bboxes = get_boxes_from_mask(filtered_img)
    bboxes = filter_bboxes(filtered_img, bboxes)
    bboxes = merge_intersection_box(bboxes)
    bboxes = filter_bboxes(filtered_img, bboxes)

    for bbox in bboxes:
        x, y, w, h = bbox
        sub_image = img[y:y+h, x:x+w]
    
        sub_image = cv2.resize(sub_image, (32, 32))
        cv2.imshow('sub_image', sub_image)
        # sub_image = np.expand_dims(sub_image, axis=0)
        
        results = model(show=False, show_labels= False, show_boxes=False, source=sub_image, verbose=False)
        name_classes_dict = results[0].names
        class_index = results[0].probs.top1
        class_conf = round(results[0].probs.top1conf.tolist(), 2)
        class_name = name_classes_dict[class_index]
        if draw is not None:
            text = f'{class_name} {class_conf}'
            cv2.rectangle(draw, (x, y), (x+w, y+h), (0, 255, 255), 4)
            cv2.putText(draw, text, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if(class_conf > 0.95 and class_name != 'unknown'):
            # print(class_name, class_conf)
            return {'class_name': class_name, 'class_conf': class_conf}
    