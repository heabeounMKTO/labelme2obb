import numpy as np

def find_xyxy(input_data):
    xyxy = input_data["shapes"][0]["points"]
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
    return np.arctan2(delta_y, delta_x) * 180.0 / np.pi

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


