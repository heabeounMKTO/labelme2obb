import cv2
import numpy as np
from common_utils import find_rect_from4, get_yolo_coords, load_json


def load_img_json_pair(img_path: str, json_path: str):
    json_data = load_json(json_path)
    img_data = cv2.imread(img_path)
    return img_data, json_data


def run():
    img , data = load_img_json_pair("dfasdvad34qwerqwef.jpeg",
                                    "dfasdvad34qwerqwef.json")
    tl, tr, bl, br = find_rect_from4(data["shapes"][0]["points"])
    yolo_cor = get_yolo_coords(img, [tl, br])
    print(yolo_cor)

if __name__ == "__main__":
    run()
