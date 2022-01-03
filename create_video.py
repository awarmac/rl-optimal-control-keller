import cv2
import numpy as np
import glob
import os
from functools import cmp_to_key
import re
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('figs_dir', type=str)
    args = parser.parse_args()
    print(args)

    img_array = []
    figs_list = glob.glob(os.path.join(args.figs_dir, "*.jpg"))
    figs_list = sorted(figs_list, 
                    key=cmp_to_key(lambda x, y: int(re.findall(r'\d+', x)[-2]) - int(re.findall(r'\d+', y)[-2])),
                    reverse=False
                )
    for filename in figs_list:
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out_filepath = os.path.join(os.path.abspath(args.figs_dir),"..", os.path.basename(os.path.dirname(args.figs_dir))+'.avi')
    print(out_filepath)
    out = cv2.VideoWriter(out_filepath, cv2.VideoWriter_fourcc(*'DIVX'), 45, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()