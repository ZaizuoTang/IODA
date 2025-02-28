import os

import cv2
import numpy as np



def Deal_mask(path,out_path,Visual):


    file_all = os.listdir(path)

    for name in file_all:

        mask_file = path + os.sep + name

        png_all_file = os.listdir(mask_file)

        leng_mask = len(png_all_file)-1

        mask_zero_path  = mask_file + os.sep + "0.png"

        mask_zero = cv2.imread(mask_zero_path,0)

        mask_blank = np.zeros_like(mask_zero).astype(np.uint16)

        Threshold = 100 #如果面积小于100的舍弃掉
        for i_png in range(leng_mask):

            name_mask = str(i_png) + ".png"
            mask_alone_path = mask_file + os.sep + name_mask
            mask_alone = cv2.imread(mask_alone_path,0)

            index = np.where(mask_alone==255)
            print(len(index[0]))

            if len(index[0])>=Threshold:
                mask_blank[index] = (i_png + 1)




        if Visual == True:

            mask_blank = mask_blank.astype(np.uint8)
            save_path = out_path + os.sep + name + '.png'
            cv2.imwrite(save_path, mask_blank)
        else:

            save_path = out_path + os.sep + name + '.npy'
            np.save(save_path,mask_blank)








if __name__ == "__main__": 


    path = "/home/tangzz/CODE/segment-anything-main/Result/"
    out_path = "/disk2/AIM_D/DATASET/SODA_DRealSR/panasonic_oneshot/Mask"
    Visual = False
    Deal_mask(path,out_path,Visual)