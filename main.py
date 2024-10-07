import cv2
import numpy as np
import json


def load_json(json_path: str):
    with open(json_path, "rb") as readfile:
        data = json.loads(readfile.read())
    return data



