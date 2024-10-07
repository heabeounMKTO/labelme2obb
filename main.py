'''
currently only single class due to need to train stuff ok
'''
import os
import cv2
import numpy as np
from common_utils import find_rect_from4, get_img_path_from_labelme, load_img_json_pair, get_file_in_folder_with_extension, get_yolo_coords, load_json, write_yolo2txt
import argparse
from pathlib import Path 


def run(folder_path: str, dest: str):
    get_all_json = get_file_in_folder_with_extension(Path(folder_path), ".json") 
    for jsonfile in get_all_json:

        img , data = load_img_json_pair(jsonfile)
        (h, w) = img.shape[:2]
        tl, tr, bl, br = find_rect_from4(data["shapes"][0]["points"])
        # yolo_cor = get_yolo_coords(img, [tl, br])
        yolo_cor = (np.array([tl,tr,br,bl]).flatten() / np.array([w, h, w, h, w, h, w, h])).tolist()
        yolo_cor.insert(0, 0)
        print(yolo_cor)
        txt_fname = os.path.join(dest,Path(jsonfile).stem + ".txt")
        img_fname = os.path.splitext(txt_fname)[0]
        img_fname = img_fname + ".jpeg" 
        # # write outputs
        cv2.imwrite(img_fname, img)
        write_yolo2txt(yolo_cor, txt_fname)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="folder to process")
    parser.add_argument("--dest", type=str, help="destination (export) folder")
    opts = parser.parse_args()
    run(opts.src, opts.dest)
