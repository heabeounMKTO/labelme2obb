import numpy as np
import json


def load_json(json_path: str):
    with open(json_path, "rb") as readfile:
        data = json.loads(readfile.read())
    return data

def find_rect_from4(input_data):
    """
    given four coordinate pairs [xy1, xy2, xy3, xy4] in no particular order, 
    returns rectangle coordinate in this order :
    `top_left, top_right, bottom_left, bottom_right`
    """
    xyxy = input_data.copy()
    xyxy = np.array(xyxy)
    top_left = xyxy[np.argmin(xyxy.sum(axis=1))]
    bottom_right = xyxy[np.argmax(xyxy.sum(axis=1))]
    diff = xyxy[0] - xyxy[1]
    top_right = xyxy[np.argmax(diff)]
    bottom_left = xyxy[np.argmin(diff)]
    return top_left, top_right, bottom_left, bottom_right

def calculate_rotation_angle(p1, p2):
    """
    get rotation angle between two points.
    takes in two points returns a float for angle
    """
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    return float(np.arctan2(delta_y, delta_x) * 180.0 / np.pi)

# stolen from yolov7 
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y

def xywh2xywhn(x, w, h):
    y = x
    y[0] = x[0] / w
    y[1] = x[1] / h
    y[2] = x[2] / w
    y[3] = x[3] / h
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y
def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x
    y[0] = w * x[0] + padw  # top left x
    y[1] = h * x[1] + padh  # top left y
    return y

def calculate_center(input_coords):
    """
    calculates the center between two points
    takes in a list of x1y1  and x2y2
    returns center
    """
    top_left = input_coords[0]
    bottom_right = input_coords[1]
    x1, y1 = top_left
    x2, y2 = bottom_right
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (cx, cy)

def get_yolo_coords(input_image: np.ndarray, input_xy_pair: list):
    (h, w) = input_image.shape[:2]
    xywh = xyxy2xywh(np.array(input_xy_pair).flatten())
    rotation_angle = calculate_rotation_angle([xywh[0], xywh[1]], input_xy_pair[1])
    xywhn = xywh2xywhn(xywh, w , h)
    xywhn = xywhn.tolist()
    xywhn.append(float(rotation_angle))
    return xywhn
